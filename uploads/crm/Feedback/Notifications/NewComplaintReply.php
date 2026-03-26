<?php

namespace Modules\Feedback\Notifications;

use App\Models\User;
use Modules\Feedback\Entities\FeedbackReply;
use Illuminate\Support\HtmlString;
use App\Notifications\BaseNotification;
use App\Models\EmailNotificationSetting;
use Illuminate\Notifications\Messages\SlackMessage;

class NewFeedbackReply extends BaseNotification
{


    /**
     * Create a new notification instance.
     *
     * @return void
     */
    private $feedback;
    private $feedbackReply;
    private $emailSetting;

    public function __construct(FeedbackReply $feedback)
    {
        $this->feedbackReply = $feedback;
        $this->feedback = $feedback->feedback;
        $this->company = $this->feedback->company;
        $this->emailSetting = EmailNotificationSetting::where('company_id', $this->company->id)->where('slug', 'new-support-ticket-request')->first();
    }

    /**
     * Get the notification's delivery channels.
     *
     * @param mixed $notifiable
     * @return array
     */
    public function via($notifiable)
    {
        $via = ['database'];

        if ($this->emailSetting->send_email == 'yes' && $notifiable->email_notifications && $notifiable->email != '') {
            array_push($via, 'mail');
        }

        if ($this->emailSetting->send_slack == 'yes' && $this->company->slackSetting->status == 'active' && $notifiable->isEmployee($notifiable->id)) {
            array_push($via, 'slack');
        }

        return $via;
    }

    public function toMail($notifiable)
    {
        $url = route('feedback.show', $this->feedback->feedback_number);
        $url = getDomainSpecificUrl($url, $this->company);


        $userReplied = FeedbackReply::orderBy('created_at', 'DESC')->first();

        if($userReplied->user_id == $notifiable->id)
        {
            $text = '<p>' . __('email.ticketReply.repliedText') . $this->feedback->feedback_number . '</p>';
        }
        else
        {
            $text = '<p>' . __('email.ticketReply.receivedText') . $this->feedback->feedback_number . '</p>';
        }

        $content = new HtmlString($text);

        return parent::build()
            ->subject(__('email.ticketReply.subject') . ' - ' . ucfirst($this->feedback->subject))
            ->markdown('mail.email', [
                'url' => $url,
                'content' => $content,
                'themeColor' => $this->company->header_color,
                'actionText' => __('email.ticketReply.action'),
                'notifiableName' => $notifiable->name
            ]);
    }

    public function toSlack($notifiable)
    {
        $slack = $notifiable->company->slackSetting;

        $message = (new SlackMessage())
            ->from(config('app.name'))
            ->image($slack->slack_logo_url);

        if (count($notifiable->employee) > 0 && (!is_null($notifiable->employee[0]->slack_username) && ($notifiable->employee[0]->slack_username != ''))) {

            return $message
                ->to('@' . $notifiable->employee[0]->slack_username)
                ->content('*' . __('email.ticketReply.subject') . '*' . "\n" . $this->feedback->subject . "\n" . __('modules.tickets.requesterName') . ' - ' . $this->feedback->requester->name . "\n" . '<' . route('tickets.show', $this->feedback->id) . '|' . __('modules.tickets.ticket') . ' #' . $this->feedback->id . '>' . "\n");
        }

        return $message->content('*' . __('email.ticketReply.subject') . '*' . "\n" .'This is a redirected notification. Add slack username for *' . $notifiable->name . '*');
    }

    /**
     * Get the array representation of the notification.
     *
     * @param mixed $notifiable
     * @return array
     */
    //phpcs:ignore
    public function toArray($notifiable)
    {
        return [
            'id' => $this->feedback->id,
            'created_at' => $this->feedbackReply->created_at->format('Y-m-d H:i:s'),
            'subject' => $this->feedback->subject,
            'user_id' => $this->feedbackReply->user_id,
            'status' => $this->feedback->status,
            'agent_id' => $this->feedback->agent_id,
            'feedback_number' => $this->feedback->feedback_number
        ];
    }

}
