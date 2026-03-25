<?php

namespace Modules\EInvoice\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Modules\EInvoice\AI\ClientInterface;
use Modules\EInvoice\Entities\EInvoiceNote;
use App\Models\Invoice;

class GenerateInvoiceNote implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $invoiceId;
    public ?int $userId;

    public function __construct(int $invoiceId, ?int $userId = null)
    {
        $this->invoiceId = $invoiceId;
        $this->userId = $userId;
    }

    public function handle(ClientInterface $ai): void
    {
        $invoice = Invoice::with(['client'])->findOrFail($this->invoiceId);

        $prompt = 'Create a concise professional note summarizing invoice #' . $invoice->id .
            ' for client ' . ($invoice->client->name ?? 'N/A') .
            ', including amount, due date, and any late fees or outstanding balance. Keep it under 120 words.';

        $content = trim($ai->complete($prompt, [
            'system' => 'You are a finance assistant that writes neutral summaries for invoices.',
            'temperature' => 0.2,
            'max_tokens' => 200,
        ]));

        EInvoiceNote::create([
            'invoice_id' => $invoice->id,
            'user_id' => $this->userId,
            'content' => $content,
        ]);
    }
}
