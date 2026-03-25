<?php

namespace Modules\Quotes\Mail;

use Illuminate\Bus\Queueable;
use Illuminate\Mail\Mailable;
use Illuminate\Queue\SerializesModels;
use Modules\Quotes\Entities\Quote;

class QuoteIssued extends Mailable
{
    use Queueable, SerializesModels;

    public Quote $quote;

    public function __construct(Quote $quote)
    {
        $this->quote = $quote;
    }

    public function build()
    {
        return $this->subject('Your Quote ' . $this->quote->number)
            ->view('quotes::quotes.email', ['quote' => $this->quote]);
    }
}
