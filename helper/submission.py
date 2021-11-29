import numpy as np
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def pred_to_submission_strings(img_id, img, patch_size):
    """Reads a single image and outputs the strings that should go into the submission file"""
    # Convert tensor to numpy array
    img_np = img.numpy()
    
    if img_np.shape[0] % patch_size != 0 or img_np.shape[1] % patch_size !=0:
        print(f"Error the patch_size{patch_size} doesn't divide both dimensions of the image ({img.shape[0]},{img.shape[1]})")
    else:
        for j in range(0, img_np.shape[1], patch_size):
            for i in range(0, img_np.shape[0], patch_size):
                patch = img_np[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield(img_id+"_{}_{},{}".format(j, i, label))


def predictions_to_submission(submission_filename, ids, predictions, patch_size=16):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for img_id, pred in zip(ids, predictions):
            f.writelines('{}\n'.format(s) for s in pred_to_submission_strings(img_id, pred, patch_size))

