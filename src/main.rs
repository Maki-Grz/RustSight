use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{core, highgui, imgproc, prelude::*, videoio};
use ort::{
    ep::ArbitrarilyConfigurableExecutionProvider,
    execution_providers::QNNExecutionProvider,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value as OrtValue,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
struct QuantParam {
    scale: f32,
    zero_point: i32,
}

#[derive(Debug, Deserialize)]
struct ModelInputOutput {
    #[serde(default)]
    quantization_parameters: Option<QuantParam>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    inputs: HashMap<String, ModelInputOutput>,
    outputs: HashMap<String, ModelInputOutput>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    model_files: HashMap<String, ModelInfo>,
}

struct Detection {
    rect: core::Rect,
    score: f32,
    class_id: u8,
}

fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let n = detections.len();
    let mut suppressed = vec![false; n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            let inter = (detections[i].rect & detections[j].rect).area();
            if inter > 0 {
                let union = (detections[i].rect.area() + detections[j].rect.area() - inter) as f32;
                if (inter as f32 / union) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    let mut idx = 0;
    detections.retain(|_| {
        let k = !suppressed[idx];
        idx += 1;
        k
    });
}

fn hwc_to_nchw(pixel_data: &[u8], h: usize, w: usize) -> Result<Array4<u8>> {
    let mut array = Array4::<u8>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            array[[0, 0, y, x]] = pixel_data[idx];
            array[[0, 1, y, x]] = pixel_data[idx + 1];
            array[[0, 2, y, x]] = pixel_data[idx + 2];
        }
    }
    Ok(array)
}

fn utc_hms() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let h = (secs % 86400) / 3600 + 2;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02} UTC", h, m, s)
}

fn main() -> Result<()> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    if let Some(ref dir) = exe_dir {
        let old_path = std::env::var("PATH").unwrap_or_default();
        let adsp_system = "C:\\Windows\\System32\\DriverStore\\FileRepository\\qcadsprpc8380.inf_arm64_a725a7f88d1ae171";
        let htp_system = "C:\\Windows\\System32\\DriverStore\\FileRepository\\qcnspmcdm8380.inf_arm64_e663a92a933cab52\\HTP";
        
        let dir_str = dir.to_string_lossy().to_string();
        
        // Configuration des chemins pour Snapdragon X
        std::env::set_var("PATH", format!("{};{};{};{}", dir_str, adsp_system, htp_system, old_path));
        std::env::set_var("ADSP_LIBRARY_PATH", format!("{};{};{}", dir_str, adsp_system, htp_system));
        
        eprintln!("[INIT] Qualcomm NPU Environment Configured");
        let dll_path = dir.join("QnnHtp.dll");
        if dll_path.exists() {
            eprintln!("[INIT] QnnHtp.dll trouvé : {}", dll_path.display());
        } else {
            eprintln!(
                "[WARN] QnnHtp.dll INTROUVABLE dans {} — le NPU ne sera pas utilisé !",
                dir.display()
            );
            eprintln!(
                "[WARN] Copier QnnHtp.dll, QnnHtpPrepare.dll, QnnHtpV73Stub.dll, QnnSystem.dll"
            );
            eprintln!("[WARN] dans le dossier de l'exe : {}", dir.display());
        }
    }

    ort::init().with_name("RustSight").commit();

    eprintln!("[INIT] ORT runtime initialisé");

    let meta_path = "gear_guard_net-onnx-w8a8/metadata.yaml";
    let meta_str = std::fs::read_to_string(meta_path).context("metadata.yaml introuvable")?;
    let meta: Metadata = serde_yaml::from_str(&meta_str)?;
    let model_meta = meta
        .model_files
        .get("gear_guard_net.onnx")
        .context("Entrée 'gear_guard_net.onnx' absente du metadata")?;

    let box_q = model_meta
        .outputs
        .get("boxes")
        .and_then(|o| o.quantization_parameters.as_ref())
        .context("Quant params 'boxes' manquants")?;

    let score_q = model_meta
        .outputs
        .get("scores")
        .and_then(|o| o.quantization_parameters.as_ref())
        .context("Quant params 'scores' manquants")?;

    let labels: Vec<String> =
        BufReader::new(std::fs::File::open("gear_guard_net-onnx-w8a8/labels.txt")?)
            .lines()
            .collect::<std::io::Result<_>>()?;

    let args: Vec<String> = std::env::args().collect();
    let input_source = args
        .get(1)
        .context("Usage: rustsight <url|chemin> [cam_id]")?;
    let cam_id = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("CAM-01")
        .to_string();

    let stream_url = if input_source.starts_with("http") {
        let out = Command::new("yt-dlp")
            .args(["-g", "-f", "best[ext=mp4]", input_source])
            .output()
            .context("yt-dlp introuvable")?;
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        input_source.clone()
    };

    let (d_w, d_h): (usize, usize) = (1280, 720);
    let (m_w, m_h): (usize, usize) = (192, 320);

    eprintln!(
        "[INIT] Résolution affichage : {}×{}  |  Modèle : {}×{}",
        d_w, d_h, m_w, m_h
    );

    let mut ffmpeg = Command::new("bin/ffmpeg.exe")
        .args([
            "-i",
            &stream_url,
            "-vf",
            &format!("scale={d_w}:{d_h}"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("ffmpeg introuvable ou impossible à démarrer")?;

    let mut stdout = ffmpeg.stdout.take().context("stdout ffmpeg inaccessible")?;

    let qnn_dll_path = exe_dir
        .as_ref()
        .map(|d| d.join("QnnHtp.dll").to_string_lossy().to_string())
        .unwrap_or_else(|| "QnnHtp.dll".to_string());

    eprintln!("[INIT] QNN backend path utilisé : {}", qnn_dll_path);

    let qnn_ep = QNNExecutionProvider::default()
        .with_backend_path(&qnn_dll_path)
        .with_arbitrary_config("backend_type", "htp")
        .with_arbitrary_config("htp_performance_mode", "burst")
        .with_arbitrary_config("htp_arch", "75")
        .with_arbitrary_config("qnn_context_priority", "high")
        .build();

    let mut session = Session::builder()
        .map_err(|e| anyhow::anyhow!("ORT builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| anyhow::anyhow!("ORT opt: {e}"))?
        .with_execution_providers([qnn_ep])
        .map_err(|e| anyhow::anyhow!("ORT EP: {e}"))?
        .commit_from_file("gear_guard_net-onnx-w8a8/gear_guard_net.onnx")
        .context("LE NPU A REFUSÉ LE MODÈLE.")?;

    eprintln!("[INIT] NPU QNN ACTIVÉ (uint8 + Optimisations OFF) !");


    // Détection du FPS via OpenCV si possible, sinon défaut 30
    let mut target_fps = 30.0;
    if !input_source.starts_with("http") {
        if let Ok(cap) = videoio::VideoCapture::from_file(&input_source, videoio::CAP_ANY) {
            if let Ok(f) = cap.get(videoio::CAP_PROP_FPS) {
                if f > 0.0 {
                    target_fps = f;
                }
            }
        }
    }
    
    let frame_interval = std::time::Duration::from_secs_f64(1.0 / target_fps);
    eprintln!("[INIT] Cible FPS : {:.2} (intervalle : {:?})", target_fps, frame_interval);

    eprintln!("[INIT] Démarrage de la boucle vidéo...\n");

    let frame_bytes = d_w * d_h * 3;
    let mut buffer = vec![0u8; frame_bytes];

    highgui::named_window("RustSight", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("RustSight", d_w as i32, d_h as i32)?;

    let mut total_detections: u64 = 0;
    let mut frame_count: u64 = 0;

    let color_green = core::Scalar::new(0.0, 220.0, 0.0, 0.0);
    let color_red = core::Scalar::new(0.0, 0.0, 220.0, 0.0);
    let color_yellow = core::Scalar::new(0.0, 200.0, 220.0, 0.0);
    let color_white = core::Scalar::new(230.0, 230.0, 230.0, 0.0);
    let color_black = core::Scalar::new(0.0, 0.0, 0.0, 200.0);

    let mut last_frame_time = std::time::Instant::now();

    loop {
        let t0 = std::time::Instant::now();
        frame_count += 1;

        if stdout.read_exact(&mut buffer).is_err() {
            eprintln!("[{}] Fin du flux après {} frames.", utc_hms(), frame_count);
            break;
        }

        // --- Limiteur de FPS RÉACTIVÉ pour usage normal ---
        let elapsed = last_frame_time.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
        last_frame_time = std::time::Instant::now();

        let frame_rgb = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                d_h as i32,
                d_w as i32,
                opencv::core::CV_8UC3,
                buffer.as_ptr() as *mut _,
                opencv::core::Mat_AUTO_STEP,
            )?
        };

        let mut resized = Mat::default();
        imgproc::resize(
            &frame_rgb,
            &mut resized,
            core::Size::new(m_w as i32, m_h as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let pixel_data = resized.data_bytes()?;
        let tensor_input = hwc_to_nchw(pixel_data, m_h, m_w)?;
        let ort_tensor = OrtValue::from_array(tensor_input)?;
        let outputs = session.run(inputs![ort_tensor])?;

        let scores_raw = outputs["scores"].try_extract_tensor::<u8>()?.1;
        let boxes_raw = outputs["boxes"].try_extract_tensor::<u8>()?.1;
        let class_raw = outputs["class_idx"].try_extract_tensor::<u8>()?.1;

        let scale_w = d_w as f32 / m_w as f32;
        let scale_h = d_h as f32 / m_h as f32;
        
        let score_scale = 0.0038347607;
        let box_scale = 1.5097413;
        let box_zp = 27i32;

        let num_anchors = scores_raw.len();
        let mut detections = Vec::with_capacity(32);

        for i in 0..num_anchors {
            let score = (scores_raw[i] as f32) * score_scale;
            if score > 0.40 {
                let label_idx = class_raw[i] as usize;
                let b = &boxes_raw[i * 4..(i + 1) * 4];
                let dq = |v: u8| (v as i32 - box_zp) as f32 * box_scale;

                let x1 = (dq(b[0]) * scale_w) as i32;
                let y1 = (dq(b[1]) * scale_h) as i32;
                let x2 = (dq(b[2]) * scale_w) as i32;
                let y2 = (dq(b[3]) * scale_h) as i32;

                let rect = core::Rect::new(
                    x1.max(0).min(d_w as i32 - 1),
                    y1.max(0).min(d_h as i32 - 1),
                    (x2 - x1).max(1).min(d_w as i32 - x1.max(0)),
                    (y2 - y1).max(1).min(d_h as i32 - y1.max(0)),
                );
                detections.push(Detection {
                    rect,
                    score,
                    class_id: class_raw[i],
                });
            }
        }

        nms(&mut detections, 0.45);
        total_detections += detections.len() as u64;

        if !detections.is_empty() {
            let ts = utc_hms();
            for det in &detections {
                let label = labels
                    .get(det.class_id as usize)
                    .map(String::as_str)
                    .unwrap_or("?");
                eprintln!(
                    "[{}] {} | {} | {:.0}% | {:?}",
                    ts,
                    cam_id,
                    label,
                    det.score * 100.0,
                    det.rect
                );
            }
        }

        let mut display = Mat::default();
        imgproc::cvt_color(
            &frame_rgb,
            &mut display,
            imgproc::COLOR_RGB2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut overlay = display.clone();
        imgproc::rectangle(
            &mut overlay,
            core::Rect::new(0, 0, d_w as i32, 52),
            core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            -1,
            8,
            0,
        )?;
        let mut temp_top = core::Mat::default();
        core::add_weighted(&overlay, 0.55, &display, 0.45, 0.0, &mut temp_top, -1)?;
        display = temp_top;

        let mut overlay2 = display.clone();
        imgproc::rectangle(
            &mut overlay2,
            core::Rect::new(0, d_h as i32 - 40, d_w as i32, 40),
            core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            -1,
            8,
            0,
        )?;
        let mut temp_bot = core::Mat::default();
        core::add_weighted(&overlay2, 0.55, &display, 0.45, 0.0, &mut temp_bot, -1)?;
        display = temp_bot;

        for det in &detections {
            let label = labels
                .get(det.class_id as usize)
                .map(String::as_str)
                .unwrap_or("?");
            let label_str = format!("{} {:.0}%", label, det.score * 100.0);

            let box_color = if det.score >= 0.80 {
                color_green
            } else {
                color_yellow
            };

            imgproc::rectangle(&mut display, det.rect, box_color, 2, 8, 0)?;

            let text_org = core::Point::new(det.rect.x, (det.rect.y - 8).max(16));
            let mut bl = 0;
            let tsz = imgproc::get_text_size(&label_str, 0, 0.6, 1, &mut bl)?;
            imgproc::rectangle(
                &mut display,
                core::Rect::new(
                    text_org.x - 2,
                    text_org.y - tsz.height - 4,
                    tsz.width + 6,
                    tsz.height + bl + 6,
                ),
                box_color,
                -1,
                8,
                0,
            )?;
            imgproc::put_text(
                &mut display,
                &label_str,
                text_org,
                0,
                0.6,
                color_black,
                1,
                16,
                false,
            )?;
        }

        let fps = 1.0 / t0.elapsed().as_secs_f64();

        let ep_label = if fps > 5.0 { "NPU" } else { "CPU?" };
        let ep_color = if fps > 5.0 { color_green } else { color_red };

        imgproc::put_text(
            &mut display,
            ep_label,
            core::Point::new(14, 36),
            0,
            0.9,
            ep_color,
            2,
            16,
            false,
        )?;

        let fps_str = format!("{:.1} FPS", fps);
        imgproc::put_text(
            &mut display,
            &fps_str,
            core::Point::new(140, 36),
            0,
            0.9,
            color_white,
            2,
            16,
            false,
        )?;

        let det_str = format!("{} detection(s)", detections.len());
        let det_color = if detections.is_empty() {
            color_white
        } else {
            color_yellow
        };
        imgproc::put_text(
            &mut display,
            &det_str,
            core::Point::new(360, 36),
            0,
            0.9,
            det_color,
            2,
            16,
            false,
        )?;

        let cam_str = format!("[ {} ]", cam_id);
        let mut bl2 = 0;
        let csz = imgproc::get_text_size(&cam_str, 0, 0.9, 2, &mut bl2)?;
        imgproc::put_text(
            &mut display,
            &cam_str,
            core::Point::new(d_w as i32 - csz.width - 14, 36),
            0,
            0.9,
            color_white,
            2,
            16,
            false,
        )?;

        let ts = utc_hms();
        imgproc::put_text(
            &mut display,
            &ts,
            core::Point::new(14, d_h as i32 - 12),
            0,
            0.75,
            color_white,
            1,
            16,
            false,
        )?;

        let total_str = format!("Total detections : {}", total_detections);
        let mut bl3 = 0;
        let tsz2 = imgproc::get_text_size(&total_str, 0, 0.65, 1, &mut bl3)?;
        imgproc::put_text(
            &mut display,
            &total_str,
            core::Point::new(d_w as i32 - tsz2.width - 14, d_h as i32 - 12),
            0,
            0.65,
            color_white,
            1,
            16,
            false,
        )?;

        highgui::imshow("RustSight", &display)?;
        if highgui::wait_key(1)? == 27 {
            break;
        }
    }

    let _ = ffmpeg.wait();
    Ok(())
}
