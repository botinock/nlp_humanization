import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from evaluate import load
from rouge_score import rouge_scorer

# Load Hugging Face's optimized ROUGE evaluator
rouge_evaluator = load("rouge")
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Sentence Transformer for cosine similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_rouge_batch(predictions, references):
    """Computes ROUGE scores in batches using Hugging Face's evaluate library."""
    return rouge_evaluator.compute(predictions=predictions, references=references)


def compute_bleu(reference, prediction):
    """Computes BLEU score for a single text."""
    return sentence_bleu([reference.split()], prediction.split(), smoothing_function=SmoothingFunction().method1)


def compute_cosine_similarity(pred, ref):
    """Computes cosine similarity using SBERT embeddings."""
    emb1 = sbert_model.encode(pred, convert_to_tensor=True)
    emb2 = sbert_model.encode(ref, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def evaluate_model(model, eval_loader, tokenizer, device, epoch, writer):
    model.eval()
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    total_bleu, total_cosine, total_loss = 0, 0, 0
    total_samples = 0
    all_references, all_candidates = [], []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            human_texts = batch["human_text"]
            LLM_texts = batch["LLM_text"]

            # Generate outputs
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
            generated_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

            # Compute loss for evaluation
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()

            # Compute ROUGE scores
            for gen_text, ref_text, ref_llm_text in zip(generated_texts, human_texts, LLM_texts):
                scores = scorer.score(gen_text, ref_text)
                for key in rouge_scores:
                    rouge_scores[key] += scores[key].fmeasure

                # Compute BLEU score
                bleu_score = sentence_bleu([ref_text.split()], gen_text.split(), smoothing_function=SmoothingFunction().method1)
                total_bleu += bleu_score

                # Compute Cosine Similarity (SBERT)
                emb1 = sbert_model.encode(gen_text, convert_to_tensor=True)
                emb2 = sbert_model.encode(ref_llm_text, convert_to_tensor=True)
                cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
                total_cosine += cosine_sim

                # Collect for BERTScore
                all_references.append(ref_llm_text)
                all_candidates.append(gen_text)

            total_samples += len(human_texts)

    # # Compute average scores
    for key in rouge_scores:
        rouge_scores[key] /= total_samples

    avg_bleu = total_bleu / total_samples
    avg_cosine = total_cosine / total_samples
    avg_loss = total_loss / total_samples

    # Compute BERTScore
    P, R, F1 = bert_score(all_candidates, all_references, lang="en")
    avg_bertscore = torch.mean(F1).item()

    # Log results to TensorBoard
    writer.add_scalar("Eval Loss", avg_loss, epoch)
    writer.add_scalar("ROUGE-1", rouge_scores["rouge1"], epoch)
    writer.add_scalar("ROUGE-2", rouge_scores["rouge2"], epoch)
    writer.add_scalar("ROUGE-L", rouge_scores["rougeL"], epoch)
    writer.add_scalar("BLEU", avg_bleu, epoch)
    writer.add_scalar("Cosine Similarity", avg_cosine, epoch)
    writer.add_scalar("BERTScore F1", avg_bertscore, epoch)

    print(f"Epoch {epoch} - Eval Loss: {avg_loss:.4f}, ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"Epoch {epoch} - BLEU: {avg_bleu:.4f}, Cosine Similarity: {avg_cosine:.4f}, BERTScore F1: {avg_bertscore:.4f}")

    return rouge_scores, avg_bleu, avg_cosine, avg_bertscore, avg_loss