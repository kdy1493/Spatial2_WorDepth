"""
간단한 annotation 시각화 스크립트
생성된 masks와 relations을 확인하기 위한 도구

python src/data_generation/visualize_annotations.py   --out_dir matched_dataset/nyu-processed-matched2/train/basement_0001a   --image_dir matched_dataset/nyu-depth-v2-matched/train/basement_0001a   --num_samples 5   --vis_dir visualizations/basement_0001a
"""
import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def visualize_sample(image_path, mask_path, rel_path, output_path):
    """단일 샘플 시각화"""
    # 이미지 & 데이터 로드
    img = np.array(Image.open(image_path))
    masks = np.load(mask_path)
    with open(rel_path) as f:
        relations = json.load(f)
    
    # Figure 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1) Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2) Masks overlay with object numbers
    mask_overlay = img.copy().astype(float) / 255.0
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(masks))))
    
    centroids = []  # 각 객체의 중심점 저장
    
    for i, mask in enumerate(masks):
        color = colors[i % 20][:3]
        # Mask를 색칠
        for c in range(3):
            mask_overlay[:, :, c] = np.where(
                mask > 0,
                mask_overlay[:, :, c] * 0.5 + color[c] * 0.5,
                mask_overlay[:, :, c]
            )
        
        # 객체 중심점 계산
        if np.any(mask > 0):
            y_coords, x_coords = np.where(mask > 0)
            cy, cx = int(y_coords.mean()), int(x_coords.mean())
            centroids.append((cx, cy, i))
    
    axes[1].imshow(mask_overlay)
    
    # 객체 번호 표시 (큰 흰색 텍스트 + 검은색 테두리)
    for cx, cy, obj_id in centroids:
        # 텍스트 테두리 (검은색)
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
            axes[1].text(cx + dx, cy + dy, str(obj_id), 
                        fontsize=12, fontweight='bold', 
                        color='black', ha='center', va='center')
        # 실제 텍스트 (흰색)
        axes[1].text(cx, cy, str(obj_id), 
                    fontsize=12, fontweight='bold', 
                    color='white', ha='center', va='center')
    
    axes[1].set_title(f"SAM Masks ({len(masks)} objects)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3) Relations text
    axes[2].axis('off')
    axes[2].set_title(f"Relations ({len(relations)} total)", fontsize=14, fontweight='bold')
    
    if len(relations) > 0:
        # Relations를 텍스트로 표시
        rel_text_lines = []
        for i, r in enumerate(relations[:25]):  # 최대 25개만
            subj = r['subject_idx']
            obj = r['object_idx']
            rel = r['relation']
            conf = r['confidence']
            
            # Emoji로 relation 표시
            if rel == 'front':
                emoji = '🔼'
            elif rel == 'behind':
                emoji = '🔽'
            elif rel == 'occludes':
                emoji = '🚫'
            elif rel == 'above':
                emoji = '⬆️'
            elif rel == 'below':
                emoji = '⬇️'
            else:
                emoji = '❓'
            
            rel_text_lines.append(
                f"{emoji} Obj{subj:2d} {rel:8s} Obj{obj:2d}  ({conf:.2f})"
            )
        
        rel_text = "\n".join(rel_text_lines)
        axes[2].text(0.05, 0.95, rel_text, 
                    fontsize=10, 
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 통계 추가
        from collections import Counter
        rel_counts = Counter(r['relation'] for r in relations)
        avg_conf = sum(r['confidence'] for r in relations) / len(relations)
        
        stats_text = f"\nStatistics:\n"
        stats_text += f"  front: {rel_counts.get('front', 0)}\n"
        stats_text += f"  behind: {rel_counts.get('behind', 0)}\n"
        stats_text += f"  occludes: {rel_counts.get('occludes', 0)}\n"
        stats_text += f"  above: {rel_counts.get('above', 0)}\n"
        stats_text += f"  below: {rel_counts.get('below', 0)}\n"
        stats_text += f"  avg conf: {avg_conf:.3f}"
        
        axes[2].text(0.05, 0.05, stats_text,
                    fontsize=10,
                    verticalalignment='bottom',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    else:
        axes[2].text(0.5, 0.5, "No relations found",
                    fontsize=14,
                    ha='center', va='center',
                    color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def compute_statistics(relations_dir):
    """전체 annotations 통계 계산"""
    all_relations = []
    file_count = 0
    
    for rel_file in glob.glob(os.path.join(relations_dir, "*_relations.json")):
        with open(rel_file) as f:
            rels = json.load(f)
            all_relations.extend(rels)
            file_count += 1
    
    if len(all_relations) == 0:
        print("No relations found!")
        return
    
    from collections import Counter
    
    print("\n" + "="*60)
    print("ANNOTATION STATISTICS")
    print("="*60)
    print(f"Total files processed: {file_count}")
    print(f"Total relations: {len(all_relations)}")
    print(f"Average relations per image: {len(all_relations)/file_count:.2f}")
    print()
    
    # Relation type 분포
    rel_types = Counter(r['relation'] for r in all_relations)
    print("Relation type distribution:")
    for rel_type, count in rel_types.most_common():
        percentage = count / len(all_relations) * 100
        print(f"  {rel_type:10s}: {count:6d} ({percentage:5.1f}%)")
    print()
    
    # Confidence 분포
    confidences = [r['confidence'] for r in all_relations]
    print("Confidence statistics:")
    print(f"  Mean: {np.mean(confidences):.3f}")
    print(f"  Std:  {np.std(confidences):.3f}")
    print(f"  Min:  {np.min(confidences):.3f}")
    print(f"  Max:  {np.max(confidences):.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print()
    
    # Confidence 히스토그램
    print("Confidence distribution:")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(confidences, bins=bins)
    for i in range(len(hist)):
        percentage = hist[i] / len(confidences) * 100
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:6d} ({percentage:5.1f}%)")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize relational annotations")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory with generated annotations")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with original images (if different from out_dir)")
    # --num_samples 인자 제거 (모든 샘플 시각화)
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    # 통계 계산
    compute_statistics(args.out_dir)
    
    # 모든 하위 폴더를 재귀적으로 탐색하여 시각화
    mask_files = sorted(glob.glob(os.path.join(args.out_dir, '**', '*_masks.npy'), recursive=True))

    if len(mask_files) == 0:
        print("No mask files found!")
        return

    print(f"Found {len(mask_files)} annotation files (recursive).")
    print(f"Visualizing all {len(mask_files)} samples...")

    for mask_file in mask_files:
        # out_dir 하위 경로를 기준으로 vis_dir 하위에 동일한 구조로 저장
        rel_dir = os.path.dirname(mask_file)
        rel_path = os.path.relpath(rel_dir, args.out_dir)
        if args.vis_dir is None:
            vis_dir = os.path.join(args.out_dir, "visualizations", rel_path)
        else:
            vis_dir = os.path.join(args.vis_dir, rel_path)
        os.makedirs(vis_dir, exist_ok=True)

        basename = os.path.basename(mask_file).replace("_masks.npy", "")
        rel_file = os.path.join(rel_dir, f"{basename}_relations.json")

        # 원본 이미지 찾기
        if args.image_dir:
            image_dir = os.path.join(args.image_dir, rel_path)
            search_dirs = [image_dir]
        else:
            search_dirs = [rel_dir, os.path.dirname(rel_dir)]

        image_file = None
        for search_dir in search_dirs:
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = os.path.join(search_dir, basename + ext)
                if os.path.exists(candidate):
                    image_file = candidate
                    break
            if image_file:
                break

        if not image_file or not os.path.exists(rel_file):
            print(f"Skipping {os.path.join(rel_path, basename)}: missing files")
            continue

        # 시각화
        output_file = os.path.join(vis_dir, f"{basename}_vis.png")
        try:
            visualize_sample(image_file, mask_file, rel_file, output_file)
        except Exception as e:
            print(f"Error visualizing {os.path.join(rel_path, basename)}: {e}")

    print(f"\nVisualization complete! Check {args.vis_dir or os.path.join(args.out_dir, 'visualizations')}/")


if __name__ == "__main__":
    main()

