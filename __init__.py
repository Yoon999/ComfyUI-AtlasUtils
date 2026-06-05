import torch
import torch.nn.functional as F

class WanReplaceFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # Wan 등으로 생성된 이미지 시퀀스 [F, H, W, C]
                "replacement_image": ("IMAGE",),  # 교체해 넣을 이미지 [1, H, W, C]
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),  # 바꾸고 싶은 프레임 번호 (0부터 시작)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_frame"
    CATEGORY = "Animation/Utils"

    def replace_frame(self, images, replacement_image, frame_index):
        # 1. 원본 이미지 시퀀스 복사 (이전 노드의 데이터를 오염시키지 않기 위함)
        output_images = images.clone()
        num_frames = output_images.shape[0]

        # 2. 인덱스 안전장치 (설정한 프레임 번호가 전체 길이를 벗어나면 마지막 프레임으로 조정)
        if frame_index >= num_frames:
            frame_index = num_frames - 1
            print(f"[ReplaceNthFrame] Warning: 입력한 인덱스가 범위를 벗어나 마지막 프레임({frame_index})으로 자동 조정되었습니다.")

        # 원본 프레임의 가로, 세로 크기 추출
        target_h, target_w = output_images.shape[1], output_images.shape[2]

        # 3. 교체용 이미지 가져오기 (배치 형태일 경우 첫 번째 이미지 선택)
        rep_img = replacement_image[0]

        # 4. 해상도가 다를 경우 자동 리사이즈 처리
        if rep_img.shape[0] != target_h or rep_img.shape[1] != target_w:
            # PyTorch 리사이즈 함수 규격에 맞게 차원 변경: [H, W, C] -> [1, C, H, W]
            rep_img_t = rep_img.permute(2, 0, 1).unsqueeze(0)
            # 원본 비디오 프레임 크기에 맞게 크기 조절
            rep_img_resized = F.interpolate(rep_img_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            # 다시 ComfyUI 표준 포맷으로 복원: [1, C, H, W] -> [H, W, C]
            rep_img = rep_img_resized.squeeze(0).permute(1, 2, 0)

        # 5. n번째 프레임 교체 실행
        output_images[frame_index] = rep_img

        return (output_images,)


class SeamlessLoopSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "offset_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("vfi_batch (Last & First)", "head_chunk", "tail_chunk")
    FUNCTION = "split_frames"
    CATEGORY = "Animation/Loop"

    def split_frames(self, images, offset_ratio):
        total_frames = images.shape[0]

        # 프레임이 충분하지 않은 경우의 예외 처리 방어 로직
        if total_frames < 3:
            print("Warning: Not enough frames to split. Returning original.")
            return (images, images, images)

        # 분할 지점(오프셋) 계산
        split_idx = int(total_frames * offset_ratio)

        # 극단적인 오프셋 방지 (각 청크에 최소 1프레임 이상 확보)
        split_idx = max(1, min(split_idx, total_frames - 2))

        # 1. VFI 배치를 위한 마지막 프레임과 첫 프레임 추출 및 결합
        # images[-1]과 images[0]은 [H, W, C]이므로 stack을 통해 [2, H, W, C]로 만듭니다.
        vfi_batch = torch.stack([images[-1], images[0]])

        # 2. 중간 덩어리 분할
        # 앞부분 덩어리: 오프셋 인덱스부터 마지막 프레임 직전까지
        head_chunk = images[split_idx:-1]

        # 뒷부분 덩어리: 첫 번째 프레임 직후부터 오프셋 인덱스 직전까지
        tail_chunk = images[1:split_idx]

        return (vfi_batch, head_chunk, tail_chunk)


class SeamlessLoopAssembler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "head_chunk": ("IMAGE",),
                "vfi_interpolated": ("IMAGE",),
                "tail_chunk": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("looped_images",)
    FUNCTION = "assemble_frames"
    CATEGORY = "Animation/Loop"

    def assemble_frames(self, head_chunk, vfi_interpolated, tail_chunk):
        # 3개의 이미지 텐서 덩어리를 배치 차원(dim=0)을 기준으로 순서대로 병합
        looped_images = torch.cat([head_chunk, vfi_interpolated, tail_chunk], dim=0)
        return (looped_images,)

# ComfyUI에 노드 등록
NODE_CLASS_MAPPINGS = {
    "WanReplaceFrame": WanReplaceFrame,
    "SeamlessLoopSplitter": SeamlessLoopSplitter,
    "SeamlessLoopAssembler": SeamlessLoopAssembler
}

# UI에 표시될 노드 이름
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanReplaceFrame": "Replace N-th Frame (Wan)",
    "SeamlessLoopSplitter": "Seamless Loop Splitter",
    "SeamlessLoopAssembler": "Seamless Loop Assembler"
}