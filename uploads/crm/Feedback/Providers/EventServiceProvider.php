<?php

namespace Modules\Feedback\Providers;
use App\Events\NewCompanyCreatedEvent;
use Modules\Feedback\Events\FeedbackEvent;
use Modules\Feedback\Events\FeedbackReplyEvent;
use Modules\Feedback\Events\FeedbackRequesterEvent;
use Modules\Feedback\Listeners\CompanyCreatedListener;
use Modules\Feedback\Listeners\FeedbackListener;
use Modules\Feedback\Listeners\FeedbackRequesterListener;
use Modules\Feedback\Listeners\FeedbackReplyListener;
use Modules\Feedback\Entities\Feedback;
use Modules\Feedback\Entities\FeedbackAgentGroups;
use Modules\Feedback\Entities\FeedbackChannel;
use Modules\Feedback\Entities\FeedbackCustomForm;
use Modules\Feedback\Entities\FeedbackEmailSetting;
use Modules\Feedback\Entities\FeedbackGroup;
use Modules\Feedback\Entities\FeedbackReply;
use Modules\Feedback\Entities\FeedbackReplyTemplate;
use Modules\Feedback\Entities\FeedbackTag;
use Modules\Feedback\Entities\FeedbackTagList;
use Modules\Feedback\Entities\FeedbackType;
use Modules\Feedback\Observers\FeedbackAgentGroupsObserver;
use Modules\Feedback\Observers\FeedbackChannelObserver;
use Modules\Feedback\Observers\FeedbackCustomFormObserver;
use Modules\Feedback\Observers\FeedbackEmailSettingObserver;
use Modules\Feedback\Observers\FeedbackGroupObserver;
use Modules\Feedback\Observers\FeedbackObserver;
use Modules\Feedback\Observers\FeedbackReplyObserver;
use Modules\Feedback\Observers\FeedbackReplyTemplateObserver;
use Modules\Feedback\Observers\FeedbackTagListObserver;
use Modules\Feedback\Observers\FeedbackTagObserver;
use Modules\Feedback\Observers\FeedbackTypeObserver;
use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;

class EventServiceProvider extends ServiceProvider
{
/**
     * The event listener mappings for the application.
     *
     * @var array
     */
    protected $listen = [
        NewCompanyCreatedEvent::class => [CompanyCreatedListener::class],
        FeedbackEvent::class => [FeedbackListener::class],
        FeedbackRequesterEvent::class => [FeedbackRequesterListener::class],
        FeedbackReplyEvent::class => [FeedbackReplyListener::class],
    ];

    protected $observers = [
        Feedback::class => [FeedbackObserver::class],
        FeedbackEmailSetting::class => [FeedbackEmailSettingObserver::class],
        FeedbackReply::class => [FeedbackReplyObserver::class],
        FeedbackReplyTemplate::class => [FeedbackReplyTemplateObserver::class],
        FeedbackGroup::class => [FeedbackGroupObserver::class],
        FeedbackAgentGroups::class => [FeedbackAgentGroupsObserver::class],
        FeedbackChannel::class => [FeedbackChannelObserver::class],
        FeedbackCustomForm::class => [FeedbackCustomFormObserver::class],
        FeedbackType::class => [FeedbackTypeObserver::class],
        FeedbackTag::class => [FeedbackTagObserver::class],
        FeedbackTagList::class => [FeedbackTagListObserver::class],
    ];
}
