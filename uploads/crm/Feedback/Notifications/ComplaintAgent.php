<?php

namespace Modules\Feedback\Notifications;

use Modules\Feedback\Entities\Feedback;
use App\Notifications\BaseNotification;
use App\Models\EmailNotificationSetting;
use Illuminate\Notifications\Messages\MailMessage;
use Illuminate\Notifications\Messages\SlackMessage;
use NotificationChannels\OneSignal\OneSignalChannel;
use NotificationChannels\OneSignal\OneSignalMessage;

class FeedbackAgent extends BaseNotification
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

        // We need to set the company in parent BaseNotification for getting proper
        // email of sender
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

        if ($this->emailSetting->send_slack == 'yes' && $this->company->slackSetting->status == 'active') {
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
        $url = route('feedback.show', $this->feedback->id);
        $url = getDomainSpecificUrl($url, $this->company);

        $content = __('email.ticketAgent.text') . '<br>' . __('modules.tickets.ticket') . ' # ' . $this->feedback->id . '<br>' . __('app.subject') . ' - ' . $this->feedback->subject;

        return parent::build()
            ->subject(__('email.ticketAgent.subject') . ' - ' . config('app.name'))
            ->markdown('mail.email', [
                'url' => $url,
                'content' => $content,
                'themeColor' => $this->company->header_color,
                'actionText' => __('email.ticketAgent.action'),
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
            'subject' => $this->feedback->subject
        ];
    }

    /**
     * Get the Slack representation of the notification.
     *
     * @param mixed $notifiable
     * @return SlackMessage
     */
    public function toSlack($notifiable)
    {
        $slack = $notifiable->company->slackSetting;

        if (count($notifiable->employee) > 0 && (!is_null($notifiable->employee[0]->slack_username) && ($notifiable->employee[0]->slack_username != ''))) {
            return (new SlackMessage())
                ->from(config('app.name'))
                ->image($slack->slack_logo_url)
                ->to('@' . $notifiable->employee[0]->slack_username)
                ->content('*' . __('email.ticketAgent.subject') . '*' . "\n" . ucfirst($this->feedback->subject) . "\n" . __('modules.tickets.requesterName') . ' - ' . mb_ucwords($this->feedback->requester->name));
        }

        return (new SlackMessage())
            ->from(config('app.name'))
            ->image($slack->slack_logo_url)
            ->content('*' . __('email.ticketAgent.subject') . '*' . "\n" .'This is a redirected notification. Add slack username for *' . $notifiable->name . '*');
    }

    // phpcs:ignore
    public function toOneSignal($notifiable)
    {
        return OneSignalMessage::create()
            ->subject(__('email.ticketAgent.subject'))
            ->body(__('email.ticketAgent.text'));
    }

}
