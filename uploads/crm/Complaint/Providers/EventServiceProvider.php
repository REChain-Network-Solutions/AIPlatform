<?php

namespace Modules\Complaint\Providers;
use App\Events\NewCompanyCreatedEvent;
use Modules\Complaint\Events\ComplaintEvent;
use Modules\Complaint\Events\ComplaintReplyEvent;
use Modules\Complaint\Events\ComplaintRequesterEvent;
use Modules\Complaint\Listeners\CompanyCreatedListener;
use Modules\Complaint\Listeners\ComplaintListener;
use Modules\Complaint\Listeners\ComplaintRequesterListener;
use Modules\Complaint\Listeners\ComplaintReplyListener;
use Modules\Complaint\Entities\Complaint;
use Modules\Complaint\Entities\ComplaintAgentGroups;
use Modules\Complaint\Entities\ComplaintChannel;
use Modules\Complaint\Entities\ComplaintCustomForm;
use Modules\Complaint\Entities\ComplaintEmailSetting;
use Modules\Complaint\Entities\ComplaintGroup;
use Modules\Complaint\Entities\ComplaintReply;
use Modules\Complaint\Entities\ComplaintReplyTemplate;
use Modules\Complaint\Entities\ComplaintTag;
use Modules\Complaint\Entities\ComplaintTagList;
use Modules\Complaint\Entities\ComplaintType;
use Modules\Complaint\Observers\ComplaintAgentGroupsObserver;
use Modules\Complaint\Observers\ComplaintChannelObserver;
use Modules\Complaint\Observers\ComplaintCustomFormObserver;
use Modules\Complaint\Observers\ComplaintEmailSettingObserver;
use Modules\Complaint\Observers\ComplaintGroupObserver;
use Modules\Complaint\Observers\ComplaintObserver;
use Modules\Complaint\Observers\ComplaintReplyObserver;
use Modules\Complaint\Observers\ComplaintReplyTemplateObserver;
use Modules\Complaint\Observers\ComplaintTagListObserver;
use Modules\Complaint\Observers\ComplaintTagObserver;
use Modules\Complaint\Observers\ComplaintTypeObserver;
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
        ComplaintEvent::class => [ComplaintListener::class],
        ComplaintRequesterEvent::class => [ComplaintRequesterListener::class],
        ComplaintReplyEvent::class => [ComplaintReplyListener::class],
    ];

    protected $observers = [
        Complaint::class => [ComplaintObserver::class],
        ComplaintEmailSetting::class => [ComplaintEmailSettingObserver::class],
        ComplaintReply::class => [ComplaintReplyObserver::class],
        ComplaintReplyTemplate::class => [ComplaintReplyTemplateObserver::class],
        ComplaintGroup::class => [ComplaintGroupObserver::class],
        ComplaintAgentGroups::class => [ComplaintAgentGroupsObserver::class],
        ComplaintChannel::class => [ComplaintChannelObserver::class],
        ComplaintCustomForm::class => [ComplaintCustomFormObserver::class],
        ComplaintType::class => [ComplaintTypeObserver::class],
        ComplaintTag::class => [ComplaintTagObserver::class],
        ComplaintTagList::class => [ComplaintTagListObserver::class],
    ];
}
