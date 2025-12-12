import os
from PIL import Image

# 경로 설정
baseline_dir = 'results/baseline_test/comparisons'
relational_dir = 'results/relational_test/comparisons'
out_dir = 'results/side_by_side_comparisons'
os.makedirs(out_dir, exist_ok=True)

# 두 폴더에서 공통으로 존재하는 파일만 대상으로 함
baseline_files = set(os.listdir(baseline_dir))
relational_files = set(os.listdir(relational_dir))
common_files = sorted(list(baseline_files & relational_files))

for fname in common_files:
    base_img = Image.open(os.path.join(baseline_dir, fname))
    rel_img = Image.open(os.path.join(relational_dir, fname))
    # 세로 크기 맞추기 (혹시 다를 경우)
    h = min(base_img.height, rel_img.height)
    base_img = base_img.resize((base_img.width, h))
    rel_img = rel_img.resize((rel_img.width, h))
    # 옆으로 붙이기
    new_img = Image.new('RGB', (base_img.width + rel_img.width, h))
    new_img.paste(base_img, (0, 0))
    new_img.paste(rel_img, (base_img.width, 0))
    new_img.save(os.path.join(out_dir, fname))

print(f"Saved {len(common_files)} side-by-side images to {out_dir}")
