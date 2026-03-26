<?php

namespace Modules\Complaint\Notifications;

use Modules\Complaint\Entities\Complaint;
use App\Notifications\BaseNotification;
use App\Models\EmailNotificationSetting;
use Illuminate\Notifications\Messages\SlackMessage;
use NotificationChannels\OneSignal\OneSignalChannel;
use NotificationChannels\OneSignal\OneSignalMessage;

class NewComplaint extends BaseNotification
{
    /**
     * Create a new notification instance.
     *
     * @return void
     */
    private $complaint;
    private $emailSetting;

    public function __construct(Complaint $complaint)
    {
        $this->complaint = $complaint;
        $this->company = $this->complaint->company;
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
     * @return \Illuminate\Notifications\Messages\MailMessage
     */
    public function toMail($notifiable)
    {
        $url = route('complaint.show', $this->complaint->complaint_number);
        $url = getDomainSpecificUrl($url, $this->company);

        $content = __('email.newTicket.text') . '<br>' . ucfirst($this->complaint->subject) . ' # ' . $this->complaint->complaint_number . '<br>' . __('modules.tickets.requesterName') . ' - ' . mb_ucwords($this->complaint->requester->name);

        return parent::build()
            ->subject(__('email.newTicket.subject') . ' - ' . ucfirst($this->complaint->subject) . ' - ' . __('modules.tickets.ticket') . ' # ' . $this->complaint->complaint_number)
            ->markdown('mail.email', [
                'url' => $url,
                'content' => $content,
                'themeColor' => $this->company->header_color,
                'actionText' => __('email.newTicket.action'),
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
            'id' => $this->complaint->id,
            'created_at' => $this->complaint->created_at->format('Y-m-d H:i:s'),
            'subject' => $this->complaint->subject,
            'user_id' => $this->complaint->user_id,
            'status' => $this->complaint->status,
            'agent_id' => $this->complaint->agent_id,
            'complaint_number' => $this->complaint->complaint_number
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
                ->content('*' . __('email.newTicket.subject') . '*' . "\n" . ucfirst($this->complaint->subject) . "\n" . __('modules.tickets.requesterName') . ' - ' . mb_ucwords($this->complaint->requester->name));
        }

        return (new SlackMessage())
            ->from(config('app.name'))
            ->image($slack->slack_logo_url)
            ->content('*' . __('email.newTicket.subject') . '*' . "\n" .'This is a redirected notification. Add slack username for *' . $notifiable->name . '*');
    }

    // phpcs:ignore
    public function toOneSignal($notifiable)
    {
        return OneSignalMessage::create()
            ->subject(__('email.newTicket.subject'))
            ->body(__('email.newTicket.text'));
    }

}
