<?php

namespace Modules\Complaint\Observers;

use Modules\Complaint\Events\ComplaintReplyEvent;
use Modules\Complaint\Entities\MailComplaintReply;
use Modules\Complaint\Entities\ComplaintEmailSetting;
use Modules\Complaint\Entities\ComplaintReply;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Mail;

class ComplaintReplyObserver
{

    public function saving(ComplaintReply $complaintReply)
    {
        if (user() && is_null($complaintReply->complaint->agent_id)) {
            $complaint = $complaintReply->complaint;
            $complaint->save();
        }
    }

    public function created(ComplaintReply $complaintReply)
    {
        $complaintReply->complaint->touch();
        $complaintEmailSetting = ComplaintEmailSetting::first();

        if (isRunningInConsoleOrSeeding()) {
            return true;
        }

        if ($complaintEmailSetting->status == 1) {
            if (!is_null($complaintReply->complaint->agent_id)) {
                if ($complaintReply->complaint->agent_id == user()->id) {
                    $toEmail = $complaintReply->complaint->client->email;

                }
                else {
                    $toEmail = $complaintReply->complaint->agent->email;
                }

                if (smtp_setting()->mail_connection == 'database') {
                    Mail::to($toEmail)->queue(new MailComplaintReply($complaintReply));

                }
                else {
                    Mail::to($toEmail)->send(new MailComplaintReply($complaintReply));
                }

            } else if(!in_array('client', user_roles())) {
                $toEmail = $complaintReply->complaint->client->email;

                if (smtp_setting()->mail_connection == 'database') {
                    Mail::to($toEmail)->queue(new MailComplaintReply($complaintReply));

                }
                else {
                    Mail::to($toEmail)->send(new MailComplaintReply($complaintReply));
                }
            }

        }

        $message = trim_editor($complaintReply->message);

        if ($message != '') {
            if (count($complaintReply->complaint->reply) > 1) {

                if (!is_null($complaintReply->complaint->agent) && user()->id != $complaintReply->complaint->agent_id) {
                    event(new ComplaintReplyEvent($complaintReply, $complaintReply->complaint->agent));
                    event(new ComplaintReplyEvent($complaintReply, $complaintReply->complaint->client));
                }
                else if (is_null($complaintReply->complaint->agent)) {
                    event(new ComplaintReplyEvent($complaintReply, null));
                    event(new ComplaintReplyEvent($complaintReply, $complaintReply->complaint->client));
                }
                else {
                    event(new ComplaintReplyEvent($complaintReply, $complaintReply->complaint->client));
                }
            }
        }

    }

}
