use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{core, highgui, imgproc, prelude::*, videoio};
use ort::{
    execution_providers::{DirectMLExecutionProvider, QNNExecutionProvider},
    inputs,
    session::Session,
    value::Value,
};
use std::env;
use std::sync::{Arc, Mutex};
use std::thread;
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
    if x2 <= x1 || y2 <= y1 { return 0.0; }
    let inter = (x2 - x1) * (y2 - y1);
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    inter as f32 / (area_a + area_b - inter) as f32
}

fn nms(boxes: Vec<core::Rect>, scores: Vec<f32>, iou_thresh: f32) -> Vec<(core::Rect, f32)> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; scores.len()];
    for &i in &order {
        if suppressed[i] { continue; }
        keep.push((boxes[i], scores[i]));
        for &j in &order {
            if suppressed[j] || j == i { continue; }
            if iou(boxes[i], boxes[j]) > iou_thresh { suppressed[j] = true; }
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
    imgproc::resize(img, &mut resized, core::Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let dx = (size.0 - nw) / 2;
    let dy = (size.1 - nh) / 2;
    let mut out = core::Mat::new_rows_cols_with_default(size.1, size.0, core::CV_8UC3, core::Scalar::new(114.0, 114.0, 114.0, 0.0))?;
    let roi = core::Rect::new(dx, dy, nw, nh);
    let mut dst = core::Mat::roi_mut(&mut out, roi)?;
    resized.copy_to(&mut dst)?;
    Ok((out, r, dx, dy))
}

fn to_tensor_ndarray(mat: &core::Mat) -> Result<Array4<f32>> {
    let mut rgb = core::Mat::default();
    imgproc::cvt_color(mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_APPROX)?;
    let mut float_mat = core::Mat::default();
    rgb.convert_to(&mut float_mat, core::CV_32F, 0.0078125, -0.99609375)?;
    let rows = float_mat.rows() as usize;
    let cols = float_mat.cols() as usize;
    let mut array = Array4::<f32>::zeros((1, 3, rows, cols));
    let mut planes = opencv::core::Vector::<core::Mat>::new();
    core::split(&float_mat, &mut planes)?;
    for i in 0..3 {
        let plane = planes.get(i)?;
        let data = plane.data_bytes()?;
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, rows * cols) };
        array.index_axis_mut(ndarray::Axis(1), i).assign(&ndarray::Array2::from_shape_vec((rows, cols), slice.to_vec())?);
    }
    Ok(array)
}

fn scrfd_infer(session: &mut Session, input: Array4<f32>) -> Result<(Vec<core::Rect>, Vec<f32>)> {
    let outputs = session.run(inputs![Value::from_array(input)?])?;
    let mut all_boxes = Vec::new();
    let mut all_scores = Vec::new();
    let strides = [8, 16, 32];
    for (i, &stride) in strides.iter().enumerate() {
        let scores = outputs[i].try_extract_tensor::<f32>()?.1;
        let bboxes = outputs[i + 3].try_extract_tensor::<f32>()?.1;
        let feat_w = 640 / stride;
        for idx in 0..scores.len() {
            if scores[idx] < 0.3 { continue; }
            let grid_idx = idx / 2;
            let row = (grid_idx / feat_w as usize) as f32;
            let col = (grid_idx % feat_w as usize) as f32;
            let (l, t, r, b) = (bboxes[idx*4]*stride as f32, bboxes[idx*4+1]*stride as f32, bboxes[idx*4+2]*stride as f32, bboxes[idx*4+3]*stride as f32);
            all_boxes.push(core::Rect::new((col*stride as f32 - l) as i32, (row*stride as f32 - t) as i32, (l+r) as i32, (t+b) as i32));
            all_scores.push(scores[idx]);
        }
    }
    Ok((all_boxes, all_scores))
}

use std::process::Command;
fn get_youtube_url(url: &str) -> Result<String> {
    let output = Command::new(".\\yt-dlp.exe").args(["-f", "bestvideo[height<=720]/best[height<=720]", "--get-url", "--no-playlist", url]).output()?;
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

#[derive(Clone)]
struct Detection { rect: core::Rect, score: f32, color: core::Scalar }

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut video_src = if args.len() > 1 && !args[1].starts_with("--") { args[1].clone() } else { "0".to_string() };
    if video_src.starts_with("http") { video_src = get_youtube_url(&video_src)?; }
    let show_fps = args.iter().any(|a| a == "--fps");

    let session = Arc::new(Mutex::new(Session::builder()?
        .with_execution_providers([QNNExecutionProvider::default().build(), DirectMLExecutionProvider::default().build()])?
        .commit_from_file("model/scrfd_500m_bnkps.onnx")?));

    let mut cam = if video_src == "0" { videoio::VideoCapture::new(0, videoio::CAP_ANY)? } else { videoio::VideoCapture::from_file(&video_src, videoio::CAP_ANY)? };
    let fps = cam.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);
    let frame_delay = if fps > 0.0 { 1000.0 / fps } else { 33.33 };

    let detections = Arc::new(Mutex::new(Vec::<Detection>::new()));
    let next_frame = Arc::new(Mutex::new(None::<core::Mat>));
    
    {
        let sess = Arc::clone(&session);
        let dets_shared = Arc::clone(&detections);
        let frame_shared = Arc::clone(&next_frame);
        thread::spawn(move || loop {
            let frame_opt = { let mut lock = frame_shared.lock().unwrap(); lock.take() };
            if let Some(frame) = frame_opt {
                if let Ok((lb, r, dx, dy)) = letterbox(&frame, (640, 640)) {
                    if let Ok(input) = to_tensor_ndarray(&lb) {
                        let mut lock = sess.lock().unwrap();
                        if let Ok((rects, scores)) = scrfd_infer(&mut lock, input) {
                            let mut results = Vec::new();
                            for (rect, score) in nms(rects, scores, 0.45) {
                                let x1 = ((rect.x as f32 - dx as f32) / r) as i32;
                                let y1 = ((rect.y as f32 - dy as f32) / r) as i32;
                                let x2 = (((rect.x + rect.width) as f32 - dx as f32) / r) as i32;
                                let y2 = (((rect.y + rect.height) as f32 - dy as f32) / r) as i32;
                                results.push(Detection { rect: core::Rect::new(x1, y1, x2-x1, y2-y1), score, color: core::Scalar::new(0.0, 255.0, 0.0, 0.0) });
                                // Heuristic: expand face box to body
                                let body_h = (y2 - y1) * 6;
                                let body_w = (x2 - x1) * 4;
                                let body_x = x1 - (body_w - (x2 - x1)) / 2;
                                let body_y = y1;
                                results.push(Detection { rect: core::Rect::new(body_x.max(0), body_y.max(0), body_w.min(frame.cols()-body_x), body_h.min(frame.rows()-body_y)), score: 0.0, color: core::Scalar::new(0.0, 0.0, 255.0, 0.0) });
                            }
                            let mut d_lock = dets_shared.lock().unwrap(); *d_lock = results;
                        }
                    }
                }
            } else { thread::sleep(Duration::from_millis(1)); }
        });
    }

    highgui::named_window("RustSight", highgui::WINDOW_NORMAL)?;
    let mut frame = core::Mat::default();
    let mut last_fps = Instant::now();
    let mut frames = 0;
    let mut fps_val = 0.0;

    loop {
        let start = Instant::now();
        if !cam.read(&mut frame)? || frame.empty() { if video_src == "0" { continue } else { break } }
        { let mut lock = next_frame.lock().unwrap(); if lock.is_none() { *lock = Some(frame.try_clone().unwrap()); } }
        
        let current_dets = { let lock = detections.lock().unwrap(); lock.clone() };
        for det in current_dets {
            imgproc::rectangle(&mut frame, det.rect, det.color, 2, imgproc::LINE_8, 0).ok();
            if det.score > 0.0 { imgproc::put_text(&mut frame, &format!("{:.0}%", det.score*100.0), core::Point::new(det.rect.x, det.rect.y-5), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, det.color, 1, imgproc::LINE_AA, false).ok(); }
        }

        frames += 1;
        if last_fps.elapsed() >= Duration::from_secs(1) { fps_val = frames as f32 / last_fps.elapsed().as_secs_f32(); frames = 0; last_fps = Instant::now(); }
        if show_fps { imgproc::put_text(&mut frame, &format!("FPS: {:.1}", fps_val), core::Point::new(10, 30), imgproc::FONT_HERSHEY_SIMPLEX, 0.7, core::Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_AA, false).ok(); }

        highgui::imshow("RustSight", &frame)?;
        let wait = (frame_delay - start.elapsed().as_millis() as f64).max(1.0) as i32;
        if highgui::wait_key(wait)? == 27 { break; }
    }
    Ok(())
}