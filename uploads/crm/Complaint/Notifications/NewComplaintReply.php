<?php

namespace Modules\Complaint\Notifications;

use App\Models\User;
use Modules\Complaint\Entities\ComplaintReply;
use Illuminate\Support\HtmlString;
use App\Notifications\BaseNotification;
use App\Models\EmailNotificationSetting;
use Illuminate\Notifications\Messages\SlackMessage;

class NewComplaintReply extends BaseNotification
{


    /**
     * Create a new notification instance.
     *
     * @return void
     */
    private $complaint;
    private $complaintReply;
    private $emailSetting;

    public function __construct(ComplaintReply $complaint)
    {
        $this->complaintReply = $complaint;
        $this->complaint = $complaint->complaint;
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

        if ($this->emailSetting->send_slack == 'yes' && $this->company->slackSetting->status == 'active' && $notifiable->isEmployee($notifiable->id)) {
            array_push($via, 'slack');
        }

        return $via;
    }

    public function toMail($notifiable)
    {
        $url = route('complaint.show', $this->complaint->complaint_number);
        $url = getDomainSpecificUrl($url, $this->company);


        $userReplied = ComplaintReply::orderBy('created_at', 'DESC')->first();

        if($userReplied->user_id == $notifiable->id)
        {
            $text = '<p>' . __('email.ticketReply.repliedText') . $this->complaint->complaint_number . '</p>';
        }
        else
        {
            $text = '<p>' . __('email.ticketReply.receivedText') . $this->complaint->complaint_number . '</p>';
        }

        $content = new HtmlString($text);

        return parent::build()
            ->subject(__('email.ticketReply.subject') . ' - ' . ucfirst($this->complaint->subject))
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
                ->content('*' . __('email.ticketReply.subject') . '*' . "\n" . $this->complaint->subject . "\n" . __('modules.tickets.requesterName') . ' - ' . $this->complaint->requester->name . "\n" . '<' . route('tickets.show', $this->complaint->id) . '|' . __('modules.tickets.ticket') . ' #' . $this->complaint->id . '>' . "\n");
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
            'id' => $this->complaint->id,
            'created_at' => $this->complaintReply->created_at->format('Y-m-d H:i:s'),
            'subject' => $this->complaint->subject,
            'user_id' => $this->complaintReply->user_id,
            'status' => $this->complaint->status,
            'agent_id' => $this->complaint->agent_id,
            'complaint_number' => $this->complaint->complaint_number
        ];
    }

}
