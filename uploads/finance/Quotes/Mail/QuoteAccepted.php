<?php

namespace Modules\Quotes\Mail;

use Illuminate\Bus\Queueable;
use Illuminate\Mail\Mailable;
use Illuminate\Queue\SerializesModels;
use Modules\Quotes\Entities\Quote;

class QuoteAccepted extends Mailable
{
    use Queueable, SerializesModels;

    public Quote $quote;
    public ?int $invoiceId;

    public function __construct(Quote $quote, ?int $invoiceId = null)
    {
        $this->quote = $quote;
        $this->invoiceId = $invoiceId;
    }

    public function build()
    {
        return $this->subject('Quote accepted: ' . $this->quote->number)
            ->view('quotes::quotes.email_accepted', [
                'quote' => $this->quote,
                'invoiceId' => $this->invoiceId,
            ]);
    }
}
