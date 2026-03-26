<?php

namespace Modules\Feedback\Observers;

use Modules\Feedback\Events\FeedbackEvent;
use Modules\Feedback\Events\FeedbackRequesterEvent;
use App\Models\Notification;
use App\Models\Order;
use Modules\Feedback\Entities\Feedback;
use App\Models\UniversalSearch;

class FeedbackObserver
{
    public function saving(Feedback $feedback)
    {
        if (!isRunningInConsoleOrSeeding()) {
            $userID = (!is_null(user())) ? user()->id : $feedback->user_id;
            $feedback->last_update_by = $userID;
        }
    }

    public function creating(Feedback $model)
    {
        if (!isRunningInConsoleOrSeeding()) {
            $userID = (!is_null(user())) ? user()->id : $model->user_id;
            $model->added_by = $userID;

            if ($model->isDirty('status') && $model->status == 'closed') {
                $model->close_date = now(company()->timezone)->format('Y-m-d');
            }

        }

        if (company()) {
            $model->company_id = company()->id;
        }

        $model->feedback_number = (int)Feedback::max('feedback_number') + 1;

    }

    public function created(Feedback $model)
    {
        if (!isRunningInConsoleOrSeeding()) {
            // Send admin notification
            event(new FeedbackEvent($model, 'NewFeedback'));

            if ($model->requester) {
                event(new FeedbackRequesterEvent($model, $model->requester));
            }

            if ($model->agent_id != '') {
                event(new FeedbackEvent($model, 'FeedbackAgent'));
            }

        }
    }

    public function updating(Feedback $feedback)
    {
        if (!isRunningInConsoleOrSeeding()) {
            if ($feedback->isDirty('status') && $feedback->status == 'closed') {
                $feedback->close_date = now(company()->timezone)->format('Y-m-d');
            }
        }
    }

    public function updated(Feedback $feedback)
    {
        if (!isRunningInConsoleOrSeeding()) {
            if ($feedback->isDirty('agent_id') && $feedback->agent_id != '') {
                event(new FeedbackEvent($feedback, 'FeedbackAgent'));
            }
        }
    }

    public function deleting(Feedback $feedback)
    {
        $universalSearches = UniversalSearch::where('searchable_id', $feedback->id)->where('module_type', 'feedback')->get();

        if ($universalSearches) {
            foreach ($universalSearches as $universalSearch) {
                UniversalSearch::destroy($universalSearch->id);
            }
        }

        $notifyData = ['Modules\Feedback\Notifications\NewFeedback', 'Modules\Feedback\Notifications\NewFeedbackReply', 'Modules\Feedback\Notifications\NewFeedbackRequester', 'Modules\Feedback\Notifications\FeedbackAgent'];

        \App\Models\Notification::deleteNotification($notifyData, $feedback->id);

    }

}
