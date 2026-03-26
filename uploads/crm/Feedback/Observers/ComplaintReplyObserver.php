<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Events\FeedbackReplyEvent;
use Modules\Feedback\Entities\MailFeedbackReply;
use Modules\Feedback\Entities\FeedbackEmailSetting;
use Modules\Feedback\Entities\FeedbackReply;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Mail;

class FeedbackReplyObserver
{

    public function saving(FeedbackReply $feedbackReply)
    {
        if (user() && is_null($feedbackReply->feedback->agent_id)) {
            $feedback = $feedbackReply->feedback;
            $feedback->save();
        }
    }

    public function created(FeedbackReply $feedbackReply)
    {
        $feedbackReply->feedback->touch();
        $feedbackEmailSetting = FeedbackEmailSetting::first();

        if (isRunningInConsoleOrSeeding()) {
            return true;
        }

        if ($feedbackEmailSetting->status == 1) {
            if (!is_null($feedbackReply->feedback->agent_id)) {
                if ($feedbackReply->feedback->agent_id == user()->id) {
                    $toEmail = $feedbackReply->feedback->client->email;

                }
                else {
                    $toEmail = $feedbackReply->feedback->agent->email;
                }

                if (smtp_setting()->mail_connection == 'database') {
                    Mail::to($toEmail)->queue(new MailFeedbackReply($feedbackReply));

                }
                else {
                    Mail::to($toEmail)->send(new MailFeedbackReply($feedbackReply));
                }

            } else if(!in_array('client', user_roles())) {
                $toEmail = $feedbackReply->feedback->client->email;

                if (smtp_setting()->mail_connection == 'database') {
                    Mail::to($toEmail)->queue(new MailFeedbackReply($feedbackReply));

                }
                else {
                    Mail::to($toEmail)->send(new MailFeedbackReply($feedbackReply));
                }
            }

        }

        $message = trim_editor($feedbackReply->message);

        if ($message != '') {
            if (count($feedbackReply->feedback->reply) > 1) {

                if (!is_null($feedbackReply->feedback->agent) && user()->id != $feedbackReply->feedback->agent_id) {
                    event(new FeedbackReplyEvent($feedbackReply, $feedbackReply->feedback->agent));
                    event(new FeedbackReplyEvent($feedbackReply, $feedbackReply->feedback->client));
                }
                else if (is_null($feedbackReply->feedback->agent)) {
                    event(new FeedbackReplyEvent($feedbackReply, null));
                    event(new FeedbackReplyEvent($feedbackReply, $feedbackReply->feedback->client));
                }
                else {
                    event(new FeedbackReplyEvent($feedbackReply, $feedbackReply->feedback->client));
                }
            }
        }

    }

}
