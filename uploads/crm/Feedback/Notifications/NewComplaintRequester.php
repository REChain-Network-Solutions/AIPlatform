<?php

namespace Modules\Feedback\Notifications;

use App\Models\SlackSetting;
use Modules\Feedback\Entities\Feedback;
use App\Notifications\BaseNotification;
use App\Models\EmailNotificationSetting;
use Illuminate\Notifications\Messages\MailMessage;
use Illuminate\Notifications\Messages\SlackMessage;
use NotificationChannels\OneSignal\OneSignalChannel;
use NotificationChannels\OneSignal\OneSignalMessage;

class NewFeedbackRequester extends BaseNotification
{


    /**
     * Create a new notification instance.
     *
     * @return void
     */
    private $feedback;
    private $emailSetting;

    public function __construct(Feedback $feedback)
    {
        $this->feedback = $feedback;
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

        if ($this->emailSetting->send_push == 'yes') {
            array_push($via, OneSignalChannel::class);
        }

        return $via;
    }

    /**
     * Get the mail representation of the notification.
     *
     * @param mixed $notifiable
     * @return MailMessage
     */
    public function toMail($notifiable): MailMessage
    {
        $url = route('feedback.show', $this->feedback->feedback_number);
        $url = getDomainSpecificUrl($url, $this->company);

        $content = __('email.newTicketRequester.text') . '<br>' . ucfirst($this->feedback->subject) . ' # ' . $this->feedback->feedback_number;

        return parent::build()
            ->subject(__('email.newTicketRequester.subject') . ' - ' . ucfirst($this->feedback->subject) . ' - ' . __('modules.tickets.ticket') . ' # ' . $this->feedback->feedback_number)
            ->markdown('mail.email', [
                'url' => $url,
                'content' => $content,
                'themeColor' => $this->company->header_color,
                'actionText' => __('email.newTicketRequester.action'),
                'notifiableName' => $notifiable->name
            ]);
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
            'created_at' => $this->feedback->created_at->format('Y-m-d H:i:s'),
            'subject' => $this->feedback->subject,
            'user_id' => $this->feedback->user_id,
            'status' => $this->feedback->status,
            'agent_id' => $this->feedback->agent_id,
            'feedback_number' => $this->feedback->feedback_number
        ];
    }

    /**
     * Get the Slack representation of the notification.
     *
     * @param mixed $notifiable
     * @return void
     */
    public function toSlack($notifiable)
    {
        $slack = $notifiable->company->slackSetting;

        if (count($notifiable->employee) > 0 && (!is_null($notifiable->employee[0]->slack_username) && ($notifiable->employee[0]->slack_username != ''))) {
            return (new SlackMessage())
                ->from(config('app.name'))
                ->image($slack->slack_logo_url)
                ->to('@' . $notifiable->employee[0]->slack_username)
                ->content('*' . __('email.newTicketRequester.subject') . '*' . "\n" . ucfirst($this->feedback->subject) . "\n" . __('modules.tickets.requesterName') . ' - ' . mb_ucwords($this->feedback->requester->name));
        }

        return (new SlackMessage())
            ->from(config('app.name'))
            ->image($slack->slack_logo_url)
            ->content('*' . __('email.newTicketRequester.subject') . '*' . "\n" .'This is a redirected notification. Add slack username for *' . $notifiable->name . '*');
    }

    // phpcs:ignore
    public function toOneSignal($notifiable)
    {
        return OneSignalMessage::create()
            ->subject(__('email.newTicketRequester.subject'))
            ->body(ucfirst($this->feedback->subject) . ' # ' . $this->feedback->id);
    }

}
