import torch


LATENT_SCALE = 0.18215


@torch.no_grad()
def encode_image_batch_to_latent(vae, image_batch, use_amp=True, chunk_size=0, latent_scale=LATENT_SCALE):
    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0 or image_batch.shape[0] <= chunk_size:
        with torch.cuda.amp.autocast(enabled=use_amp):
            return vae.encode(image_batch).latent_dist.sample().mul_(latent_scale)

    latent_chunks = []
    for start_idx in range(0, image_batch.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, image_batch.shape[0])
        with torch.cuda.amp.autocast(enabled=use_amp):
            latent_chunk = vae.encode(image_batch[start_idx:end_idx]).latent_dist.sample().mul_(latent_scale)
        latent_chunks.append(latent_chunk)
    return torch.cat(latent_chunks, dim=0)


@torch.no_grad()
def decode_latent_batch_to_image(vae, latent_batch, use_amp=True, chunk_size=0, latent_scale=LATENT_SCALE):
    chunk_size = int(chunk_size or 0)
    latent_batch = latent_batch / latent_scale
    if chunk_size <= 0 or latent_batch.shape[0] <= chunk_size:
        with torch.cuda.amp.autocast(enabled=use_amp):
            return vae.decode(latent_batch).sample

    image_chunks = []
    for start_idx in range(0, latent_batch.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, latent_batch.shape[0])
        with torch.cuda.amp.autocast(enabled=use_amp):
            image_chunk = vae.decode(latent_batch[start_idx:end_idx]).sample
        image_chunks.append(image_chunk)
    return torch.cat(image_chunks, dim=0)
