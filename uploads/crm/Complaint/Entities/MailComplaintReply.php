<?php

namespace Modules\Complaint\Entities;

use Modules\Complaint\Entities\ComplaintEmailSetting;
use Modules\Complaint\Entities\ComplaintReply as ModelsComplaintReply;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Mail\Mailable;
use Illuminate\Queue\SerializesModels;

class MailComplaintReply extends Mailable implements ShouldQueue
{

    use Queueable, SerializesModels;

    private $complaintEmailSetting;
    public $complaintReply;

    /**
     * Create a new message instance.
     *
     * @return void
     */
    public function __construct(ModelsComplaintReply $complaintReply)
    {
        $this->complaintEmailSetting = ComplaintEmailSetting::first();
        $this->complaintReply = $complaintReply;
    }

    /**
     * Build the message.
     *
     * @return $this
     */
    public function build()
    {
        $previousReply = ModelsComplaintReply::where('complaint_id', $this->complaintReply->complaint_id)
            ->whereNotNull('imap_message_id')->orderBy('id', 'desc')
            ->first();

        if ($this->complaintEmailSetting->status == 1) {
            $this->from($this->complaintEmailSetting->mail_from_email, $this->complaintEmailSetting->mail_from_name)
                ->subject($this->complaintReply->complaint->subject)
                ->view('complaint::emails.reply');

            if (!is_null($previousReply) && !is_null($previousReply->imap_message_id)) {
                $this->withSwiftMessage(function ($message) use ($previousReply) {
                    $message->getHeaders()->addTextHeader(
                        'In-Reply-To', '<' . $previousReply->imap_message_id . '>'
                    );

                    ModelsComplaintReply::where('id', $this->complaintReply->id)->update(['imap_message_id' => $message->getId()]);
                });
            }

            return $this;
        }
    }

}
