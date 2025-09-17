
# OCR Losses Usage (DB + CTC)

## Detection (DB)
from ocr.losses import DBLossLite

criterion = DBLossLite(bce_ratio=0.5, dice_ratio=0.5, l1_ratio=0.0)
out = det_model(images)                        # {'prob_map': (B,1,H,W), ...}
loss_dict = criterion(out['prob_map'], gt_prob=prob_gt, mask=prob_mask)
loss = loss_dict['loss']

- `prob_gt`: 0/1 map for shrunken text regions (float tensor).
- `prob_mask`: 1 for valid pixels, 0 to ignore (e.g., 'do not care' regions).
- If you also predict a threshold map, pass `thresh_map` and `gt_thresh` and set `l1_ratio>0`.

## Recognition (CTC)
from ocr.losses import CTCLossWrapper

criterion = CTCLossWrapper(blank=0)           # keep blank_id consistent with your charset
out = rec_model(crops)                         # {'logits': (B,T,K)}
targets = batch_text_ids                       # list[list[int]], without blank
loss_dict = criterion(out['logits'], targets)
loss = loss_dict['loss']
