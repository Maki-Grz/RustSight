use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{core, highgui, imgproc, prelude::*, videoio};
use ort::{inputs, session::Session, value::Value};
use std::env;
use std::time::{Duration, Instant};

fn parse_arg_value(args: &[String], key: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == key {
            if i + 1 < args.len() {
                return args[i + 1].clone();
            }
        } else if args[i].starts_with(&(key.to_string() + "=")) {
            return args[i][key.len() + 1..].to_string();
        }
    }
    default.to_string()
}

fn iou(a: core::Rect, b: core::Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let inter = (x2 - x1) * (y2 - y1);
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    inter as f32 / (area_a + area_b - inter) as f32
}

fn nms(boxes: Vec<core::Rect>, scores: Vec<f32>, iou_thresh: f32) -> Vec<(core::Rect, f32)> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<(core::Rect, f32)> = Vec::new();
    let mut suppressed = vec![false; scores.len()];
    for &i in &order {
        if suppressed[i] {
            continue;
        }
        let bi = boxes[i];
        let si = scores[i];
        keep.push((bi, si));
        for &j in &order {
            if suppressed[j] {
                continue;
            }
            if j == i {
                continue;
            }
            if iou(bi, boxes[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn letterbox(img: &core::Mat, size: (i32, i32)) -> Result<(core::Mat, f32, i32, i32)> {
    let h = img.rows();
    let w = img.cols();
    let r = (size.1 as f32 / h as f32).min(size.0 as f32 / w as f32);
    let nw = (w as f32 * r).round() as i32;
    let nh = (h as f32 * r).round() as i32;
    let mut resized = core::Mat::default();
    imgproc::resize(
        img,
        &mut resized,
        core::Size::new(nw, nh),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let dx = (size.0 - nw) / 2;
    let dy = (size.1 - nh) / 2;
    let mut out = core::Mat::new_rows_cols_with_default(
        size.1,
        size.0,
        core::CV_8UC3,
        core::Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;
    let roi = core::Rect::new(dx, dy, nw, nh);
    let mut dst = core::Mat::roi_mut(&mut out, roi)?;
    resized.copy_to(&mut dst)?;
    Ok((out, r, dx, dy))
}

fn to_tensor_ndarray(mat: &core::Mat) -> Result<Array4<f32>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut rgb = core::Mat::default();
    imgproc::cvt_color(
        mat,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        core::AlgorithmHint::ALGO_HINT_APPROX,
    )?;
    let data = rgb.data_bytes()?;

    let mut array = Array4::<f32>::zeros((1, 3, rows as usize, cols as usize));
    for y in 0..rows as usize {
        for x in 0..cols as usize {
            let idx = (y * cols as usize + x) * 3;
            // Normalisation SCRFD standard: (x - 127.5) / 128.0
            array[[0, 0, y, x]] = (data[idx] as f32 - 127.5) / 128.0;
            array[[0, 1, y, x]] = (data[idx + 1] as f32 - 127.5) / 128.0;
            array[[0, 2, y, x]] = (data[idx + 2] as f32 - 127.5) / 128.0;
        }
    }
    Ok(array)
}

fn scrfd_infer(session: &mut Session, input: Array4<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
    let input_tensor = Value::from_array(input)?;
    let outputs = session.run(inputs![input_tensor])?;

    let mut all_boxes = Vec::new();
    let mut all_scores = Vec::new();

    let strides = [8, 16, 32];
    // SCRFD Output indices:
    // Scores: 0 (s8), 1 (s16), 2 (s32)
    // BBoxes: 3 (b8), 4 (b16), 5 (b32)

    for (i, &stride) in strides.iter().enumerate() {
        let scores_view = outputs[i].try_extract_tensor::<f32>()?;
        let bboxes_view = outputs[i + 3].try_extract_tensor::<f32>()?;

        let scores = scores_view.1;
        let bboxes = bboxes_view.1;

        // La dimension attendue est [1, num_anchors * H * W, 1] ou similaire
        // Pour SCRFD, il y a généralement 2 ancres par position
        let num_points = scores.len();
        let num_anchors = 2;

        let feat_w = 640 / stride;

        for idx in 0..num_points {
            let score = scores[idx];
            if score < 0.3 {
                continue;
            }

            let anchor_idx = idx % num_anchors;
            let grid_idx = idx / num_anchors;
            let row = (grid_idx / feat_w as usize) as f32;
            let col = (grid_idx % feat_w as usize) as f32;

            let b_offset = idx * 4;
            let l = bboxes[b_offset + 0] * stride as f32;
            let t = bboxes[b_offset + 1] * stride as f32;
            let r = bboxes[b_offset + 2] * stride as f32;
            let b = bboxes[b_offset + 3] * stride as f32;

            let cx = col * stride as f32;
            let cy = row * stride as f32;

            all_boxes.push(cx - l);
            all_boxes.push(cy - t);
            all_boxes.push(cx + r);
            all_boxes.push(cy + b);
            all_scores.push(score);
        }
    }

    Ok((all_boxes, all_scores))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let video_src = if args.len() > 1 {
        if args[1].starts_with("--") {
            "0".to_string()
        } else {
            args[1].clone()
        }
    } else {
        "0".to_string()
    };
    let width_opt: i32 = parse_arg_value(&args, "--width", "0").parse().unwrap_or(0);
    let height_opt: i32 = parse_arg_value(&args, "--height", "0").parse().unwrap_or(0);
    let conf_threshold: f32 = parse_arg_value(&args, "--confidence", "0.5")
        .parse()
        .unwrap_or(0.5);
    let show_fps = args.iter().any(|a| a == "--fps");
    let model_path = parse_arg_value(&args, "--model", "model/scrfd_500m_bnkps.onnx");

    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    println!("Model Loaded. Detection started...");

    let mut cam = if video_src == "0" {
        let mut c = videoio::VideoCapture::new(0, videoio::CAP_ANY).context("camera")?;
        if width_opt > 0 {
            let _ = c.set(videoio::CAP_PROP_FRAME_WIDTH, width_opt as f64);
        }
        if height_opt > 0 {
            let _ = c.set(videoio::CAP_PROP_FRAME_HEIGHT, height_opt as f64);
        }
        let _ = c.set(videoio::CAP_PROP_BUFFERSIZE, 1.0);
        c
    } else {
        let mut c = videoio::VideoCapture::from_file(&video_src, videoio::CAP_ANY)
            .with_context(|| format!("video {}", video_src))?;
        let _ = c.set(videoio::CAP_PROP_BUFFERSIZE, 1.0);
        c
    };
    if !videoio::VideoCapture::is_opened(&cam)? {
        anyhow::bail!("source not opened")
    }

    let window = "RustSight - SCRFD Multi-Scale";
    highgui::named_window(window, highgui::WINDOW_NORMAL).context("window")?;
    let mut frame = core::Mat::default();
    let mut fps_last = Instant::now();
    let mut fps_frames: u32 = 0;
    let mut fps_value: f32 = 0.0;

    loop {
        if !cam.read(&mut frame)? || frame.empty() {
            if video_src == "0" { continue } else { break }
        }

        let frame_width = frame.cols();
        let frame_height = frame.rows();

        let (lb, r, dx, dy) = letterbox(&frame, (640, 640))?;
        let input = to_tensor_ndarray(&lb)?;
        let (boxes, scores) = scrfd_infer(&mut session, input)?;

        let mut rects: Vec<core::Rect> = Vec::new();
        let mut confs: Vec<f32> = Vec::new();

        let n = scores.len();
        for i in 0..n {
            let s = scores[i];
            if s < conf_threshold {
                continue;
            }
            let x1 = boxes[i * 4 + 0];
            let y1 = boxes[i * 4 + 1];
            let x2 = boxes[i * 4 + 2];
            let y2 = boxes[i * 4 + 3];

            let mut xi1 = ((x1 - dx as f32) / r).round() as i32;
            let mut yi1 = ((y1 - dy as f32) / r).round() as i32;
            let mut xi2 = ((x2 - dx as f32) / r).round() as i32;
            let mut yi2 = ((y2 - dy as f32) / r).round() as i32;

            xi1 = xi1.max(0);
            yi1 = yi1.max(0);
            xi2 = xi2.min(frame_width - 1);
            yi2 = yi2.min(frame_height - 1);

            if xi2 <= xi1 || yi2 <= yi1 {
                continue;
            }
            rects.push(core::Rect::new(xi1, yi1, xi2 - xi1, yi2 - yi1));
            confs.push(s);
        }

        let dets = nms(rects, confs, 0.45);
        for (face_rect, score) in dets {
            imgproc::rectangle(
                &mut frame,
                face_rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )
            .ok();
            let label = format!("{:.2}%", score * 100.0);
            imgproc::put_text(
                &mut frame,
                &label,
                core::Point::new(face_rect.x, (face_rect.y - 10).max(0)),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.45,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )
            .ok();
        }

        fps_frames += 1;
        let elapsed = fps_last.elapsed();
        if elapsed >= Duration::from_secs(1) {
            fps_value = fps_frames as f32 / elapsed.as_secs_f32();
            fps_frames = 0;
            fps_last = Instant::now();
        }
        if show_fps {
            let fps_text = format!("FPS: {:.1}", fps_value);
            imgproc::put_text(
                &mut frame,
                &fps_text,
                core::Point::new(10, 20),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )
            .ok();
        }

        highgui::imshow(window, &frame).context("imshow")?;
        let key = highgui::wait_key(1).context("key")?;
        if key == 27 || (key != -1 && (key as u8 as char == 'q' || key as u8 as char == 'Q')) {
            break;
        }
    }

    Ok(())
}
