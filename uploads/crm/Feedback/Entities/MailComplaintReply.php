<?php

namespace Modules\Feedback\Entities;

use Modules\Feedback\Entities\FeedbackEmailSetting;
use Modules\Feedback\Entities\FeedbackReply as ModelsFeedbackReply;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Mail\Mailable;
use Illuminate\Queue\SerializesModels;

class MailFeedbackReply extends Mailable implements ShouldQueue
{

    use Queueable, SerializesModels;

    private $feedbackEmailSetting;
    public $feedbackReply;

    /**
     * Create a new message instance.
     *
     * @return void
     */
    public function __construct(ModelsFeedbackReply $feedbackReply)
    {
        $this->feedbackEmailSetting = FeedbackEmailSetting::first();
        $this->feedbackReply = $feedbackReply;
    }

    /**
     * Build the message.
     *
     * @return $this
     */
    public function build()
    {
        $previousReply = ModelsFeedbackReply::where('feedback_id', $this->feedbackReply->feedback_id)
            ->whereNotNull('imap_message_id')->orderBy('id', 'desc')
            ->first();

        if ($this->feedbackEmailSetting->status == 1) {
            $this->from($this->feedbackEmailSetting->mail_from_email, $this->feedbackEmailSetting->mail_from_name)
                ->subject($this->feedbackReply->feedback->subject)
                ->view('feedback::emails.reply');

            if (!is_null($previousReply) && !is_null($previousReply->imap_message_id)) {
                $this->withSwiftMessage(function ($message) use ($previousReply) {
                    $message->getHeaders()->addTextHeader(
                        'In-Reply-To', '<' . $previousReply->imap_message_id . '>'
                    );

                    ModelsFeedbackReply::where('id', $this->feedbackReply->id)->update(['imap_message_id' => $message->getId()]);
                });
            }

            return $this;
        }
    }

}
