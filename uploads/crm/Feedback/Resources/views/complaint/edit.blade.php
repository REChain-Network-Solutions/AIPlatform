@extends('layouts.app')

@push('styles')
    <link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
    <link rel="stylesheet" href="{{ asset('vendor/css/tagify.css') }}">

    <style>
        .message-action {
            visibility: hidden;
        }

        .ticket-left .card:hover .message-action {
            visibility: visible;
        }

        .file-action {
            visibility: hidden;
        }

        .file-card:hover .file-action {
            visibility: visible;
        }

        .frappe-chart .chart-legend {
            display: none;
        }
    </style>
    <script src="{{ asset('vendor/jquery/frappe-charts.min.iife.js') }}"></script>
    <script src="{{ asset('vendor/jquery/Chart.min.js') }}"></script>
@endpush

@section('filter-section')
    <!-- TICKET HEADER START -->
    <div class="d-flex px-4 filter-box bg-white">
        <a href="javascript:;"
            class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey pr-3 reply-button"><i
                class="fa fa-reply mr-0 mr-lg-2 mr-md-2"></i><span
                class="d-none d-lg-block d-md-block">@lang('app.reply')</span>
        </a>

        @if ($viewFeedbackPermission == 'all' || ($viewFeedbackPermission == 'owned' && $feedback->agent_id == user()->id))
            <div id="ticket-closed" @if ($feedback->status == 'closed') style="display:none" @endif>
                <a href="javascript:;" data-status="closed"
                    class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey px-3 submit-ticket"><i
                        class="fa fa-times-circle mr-0 mr-lg-2 mr-md-2"></i><span
                        class="d-none d-lg-block d-md-block">@lang('app.close')</span></a>
            </div>

            <a href="javascript:;"
                class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey px-3 delete-ticket"><i
                    class="fa fa-trash mr-0 mr-lg-2 mr-md-2"></i><span
                    class="d-none d-lg-block d-md-block">@lang('app.delete')</span>
            </a>
        @endif

        <a onclick="openTicketsSidebar()"
            class="d-flex d-lg-none ml-auto align-items-center justify-content-center height-44 text-dark-grey text-capitalize border-left-grey pl-3"><i
                class="fa fa-ellipsis-v"></i>
        </a>
    </div>
    <!-- TICKET HEADER END -->
@endsection

@section('content')
    <!-- TICKET START -->
    <div class="ticket-wrapper bg-white border-top-0 d-lg-flex">

        <!-- TICKET LEFT START -->
        <div class="ticket-left w-100">
            <x-form id="updateTicket2" method="PUT">
                <input type="hidden" name="status" id="status" value="{{ $feedback->status }}">
                <input type="hidden" id="feedback_reply_id" value="">

                <!-- START -->
                <div class="d-flex justify-content-between align-items-center p-3 border-right-grey border-bottom-grey">
                    <span>
                        <p class="f-15 f-w-500 mb-0">{{ $feedback->subject }}</p>
                        <p class="f-11 text-lightest mb-0">@lang('modules.tickets.requestedOn')
                            {{ $feedback->created_at->timezone(company()->timezone)->translatedFormat(company()->date_format . ' ' . company()->time_format) }}
                        </p>
                    </span>
                    <span>
                        @if ($feedback->status == 'open')
                            @php
                                $statusColor = 'red';
                            @endphp
                        @elseif($feedback->status == 'pending')
                            @php
                                $statusColor = 'yellow';
                            @endphp
                        @elseif($feedback->status == 'resolved')
                            @php
                                $statusColor = 'dark-green';
                            @endphp
                        @elseif($feedback->status == 'closed')
                            @php
                                $statusColor = 'blue';
                            @endphp
                        @endif
                        <p class="mb-0 text-capitalize ticket-status">
                            <x-status :color="$statusColor" :value="__('app.' . $feedback->status)" />
                        </p>
                    </span>
                </div>
                <!-- END -->
                <!-- TICKET MESSAGE START -->
                <div class="ticket-msg border-right-grey" data-menu-vertical="1" data-menu-scroll="1"
                    data-menu-dropdown-timeout="500" id="ticketMsg">

                    @foreach ($feedback->reply as $reply)
                        <x-cards.ticket :message="$reply" :user="$reply->user" />
                    @endforeach

                </div>
                <!-- TICKET MESSAGE END -->

                <div class="col-md-12 border-top border-right d-none mb-5" id="reply-section">
                    <div class="form-group my-3">
                        @if ($feedback->requester->id != user()->id || (!is_null($feedback->agent_id) && $feedback->agent_id != user()->id))
                            <p class="f-w-500">
                                @lang('app.to'):
                                {{ $feedback->requester->id != user()->id ? $feedback->requester->name : $feedback->agent->name }}
                            </p>
                        @endif
                        <div id="description"></div>
                        <textarea name="message" id="description-text" class="d-none"></textarea>
                        <input type="hidden" value="{{ implode(',', user_roles()) }}" id="role">
                    </div>
                    <div class="my-3">
                        <a class="f-15 f-w-500" href="javascript:;" id="add-file"><i
                                class="fa fa-paperclip font-weight-bold mr-1"></i>@lang('modules.projects.uploadFile')</a>
                    </div>
                    <x-forms.file-multiple class="mr-0 mr-lg-2 mr-md-2 upload-section d-none" fieldLabel=""
                        fieldName="file[]" fieldId="task-file-upload-dropzone" />
                </div>

                <div class="ticket-reply-back justify-content-start px-lg-4 px-md-4 px-3 py-3  d-flex bg-white border-top-grey border-right-grey"
                    id="reply-section-action">

                    <x-forms.button-primary class="reply-button mr-3" icon="reply">@lang('app.reply')
                    </x-forms.button-primary>

                    <x-forms.link-secondary :link="route('feedback.index')" icon="arrow-left">@lang('app.back')
                    </x-forms.link-secondary>
                </div>
                <div class="ticket-reply-back flex-row justify-content-start px-lg-4 px-md-4 px-3 py-3 c-inv-btns bg-white border-top-grey border-right-grey d-none"
                    id="reply-section-action-2">
                    @if ($viewFeedbackPermission == 'all' || $feedback->agent_id == user()->id)
                        <div class="inv-action dropup mr-3">
                            <button class="btn-primary dropdown-toggle" type="button" data-toggle="dropdown"
                                aria-haspopup="true" aria-expanded="false">
                                @lang('app.submit')
                                <span><i class="fa fa-chevron-up f-15 text-white"></i></span>
                            </button>
                            <!-- DROPDOWN - INFORMATION -->
                            <ul class="dropdown-menu" aria-labelledby="dropdownMenuBtn" tabindex="0">
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="open">
                                        <x-status color="red" :value="__('modules.tickets.submitOpen')" />
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="pending">
                                        <x-status color="yellow" :value="__('modules.tickets.submitPending')" />
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="resolved">
                                        <x-status color="dark-green" :value="__('modules.tickets.submitResolved')" />
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="closed">
                                        <x-status color="blue" :value="__('modules.tickets.submitClosed')" />
                                    </a>
                                </li>

                            </ul>
                        </div>
                    @else
                        <x-forms.button-primary icon="check" data-status="open" class="submit-ticket mr-3">
                            @lang('app.submit')
                        </x-forms.button-primary>
                    @endif

                    @if (!in_array('client', user_roles()))
                        <div class="inv-action dropup mr-3">
                            <button class="btn-secondary dropdown-toggle" type="button" data-toggle="dropdown"
                                aria-haspopup="true" aria-expanded="false">
                                <i class="fa fa-bolt f-15 mr-1"></i>
                                @lang('modules.tickets.applyTemplate')
                                <span><i class="fa fa-chevron-up f-15"></i></span>
                            </button>
                            <!-- DROPDOWN - INFORMATION -->
                            <ul class="dropdown-menu" aria-labelledby="dropdownMenuBtn" tabindex="0">
                                @forelse($templates as $template)
                                    <li><a href="javascript:;" data-template-id="{{ $template->id }}"
                                            class="dropdown-item f-14 text-dark apply-template">{{ ucfirst($template->reply_heading) }}</a>
                                    </li>
                                @empty
                                    <li><a class="dropdown-item f-14 text-dark">@lang('messages.noTemplateFound')</a></li>
                                @endforelse
                            </ul>
                        </div>
                    @endif

                    <x-forms.link-secondary id="cancel-reply" class="border-0" link="javascript:;">@lang('app.cancel')
                    </x-forms.link-secondary>
                </div>
            </x-form>
        </div>
        <!-- TICKET LEFT END -->
        @if ($viewFeedbackPermission == 'all')
            <!-- TICKET RIGHT START -->
            <div class="mobile-close-overlay w-100 h-100" id="close-tickets-overlay"></div>
            <div class="ticket-right bg-white" id="ticket-detail-contact">
                <a class="d-block d-lg-none close-it" id="close-tickets"><i class="fa fa-times"></i></a>
                <div id="tabs">
                    <nav class="tabs px-2 border-bottom-grey">
                        <div class="nav" id="nav-tab" role="tablist">
                            <a class="nav-item nav-link f-14 active" id="nav-detail-tab" data-toggle="tab"
                                href="#nav-details" role="tab" aria-controls="nav-email"
                                aria-selected="false">@lang('app.details')</a>
                            <a class="nav-item nav-link f-14" id="nav-contact-tab" data-toggle="tab" href="#nav-contact"
                                role="tab" aria-controls="nav-slack" aria-selected="true">@lang('app.contact')</a>
                            <a class="nav-item nav-link f-14" id="nav-other-tab" data-toggle="tab" href="#nav-other"
                                role="tab">@lang('app.other')</a>
                        </div>
                    </nav>
                </div>
                <div class="tab-content" id="nav-tabContent">
                    <!-- DETAILS START -->
                    <div class="tab-pane fade show active" id="nav-details" role="tabpanel"
                        aria-labelledby="nav-detail-tab">
                        <x-form id="updateTicket1">
                            <!-- TICKET FILTERS START -->
                            <div class="ticket-filters p-4 w-100 position-relative border-bottom">
                                <div class="more-filter-items">
                                    <x-forms.label class="my-3" fieldId="tags" :fieldLabel="__('engineerings::modules.wr')">
                                    </x-forms.label><br>
                                    @if ($wr->isEmpty())
                                        <a href="{{ route('feedback.createWR', [$feedback->id, $feedback->feedback_number]) }}"
                                            class="btn btn-outline-secondary border-grey">@lang('app.add')
                                            @lang('engineerings::modules.wr')</a>
                                    @else
                                        <div class="r-t-items d-flex">
                                            <div class="r-t-items-right ">
                                                <span class="d-flex mb-1">
                                                    <span class="mr-3 f-w-500 text-dark-grey"><a class="text-dark"
                                                            href="{{ route('engineerings.show', $wr[0]->id) }}">#{{ $wr[0]->wr_no }}</a></span>
                                                    @if ($wr[0]->status == 'unchecked')
                                                        @php
                                                            $statusColor = 'red';
                                                        @endphp
                                                    @elseif($wr[0]->status == 'checked')
                                                        @php
                                                            $statusColor = 'blue';
                                                        @endphp
                                                    @endif
                                                    <span class="f-13 text-darkest-grey text-capitalize">
                                                        <x-status :color="$statusColor" :value="$wr[0]->status" />
                                                    </span>
                                                </span>
                                            </div>
                                        </div>
                                    @endif
                                </div>
                                @if (!$wr->isEmpty())
                                    <div class="more-filter-items">
                                        <x-forms.label class="my-3" fieldId="tags" :fieldLabel="__('engineerings::modules.wo')">
                                        </x-forms.label><br>
                                        @if (!$wr->isEmpty() && $wo->isEmpty())
                                            <a href="{{ route('feedback.createWO', [$feedback->id, $feedback->feedback_number, $wr[0]->id]) }}"
                                                class="btn btn-outline-secondary border-grey">@lang('app.add')
                                                @lang('engineerings::modules.wo')</a>
                                        @endif
                                        @if (!$wo->isEmpty())
                                            <div class="r-t-items d-flex">
                                                <div class="r-t-items-right ">
                                                    <span class="d-flex mb-1">
                                                        <span class="mr-3 f-w-500 text-dark-grey"><a class="text-dark"
                                                                href="{{ route('work.show', $wo[0]->id) }}">#{{ $wo[0]->nomor_wo }}</a></span>
                                                        @if ($wo[0]->status == 'incomplete')
                                                            @php
                                                                $statusColor = 'red';
                                                            @endphp
                                                        @elseif($wo[0]->status == 'pending')
                                                            @php
                                                                $statusColor = 'yellow';
                                                            @endphp
                                                        @elseif($wo[0]->status == 'completed')
                                                            @php
                                                                $statusColor = 'blue';
                                                            @endphp
                                                        @endif
                                                        <span class="f-13 text-darkest-grey text-capitalize">
                                                            <x-status :color="$statusColor" :value="$wo[0]->status" />
                                                        </span>
                                                    </span>
                                                </div>
                                            </div>
                                        @endif
                                    </div>
                                @endif
                                <div class="more-filter-items">
                                    <x-forms.label class="my-3" fieldId="agent_id" :fieldLabel="__('modules.tickets.agent')">
                                    </x-forms.label>
                                    <x-forms.input-group>
                                        <select class="form-control select-picker " name="agent_id" id="agent_id"
                                            data-live-search="true" data-container="body" data-size="8">
                                            <option value="">--</option>
                                            @foreach ($groups as $group)
                                                @if (count($group->enabledAgents) > 0)
                                                    <optgroup label="{{ mb_ucwords($group->group_name) }}">
                                                        @foreach ($group->enabledAgents as $agent)
                                                            <x-user-option :user="$agent->user"
                                                                :selected="$agent->user->id == $feedback->agent_id"></x-user-option>
                                                        @endforeach
                                                    </optgroup>
                                                @endif
                                            @endforeach
                                        </select>
                                        @if ($viewFeedbackPermission == 'all')
                                            <x-slot name="append">
                                                <button id="addAgent" type="button"
                                                    class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                            </x-slot>
                                        @endif
                                    </x-forms.input-group>
                                </div>
                                <div class="more-filter-items">
                                    <x-forms.select fieldId="priority" :fieldLabel="__('modules.tasks.priority')" fieldName="priority"
                                        data-container="body">
                                        <option @if ($feedback->priority == 'low') selected @endif value="low">
                                            @lang('app.low')</option>
                                        <option @if ($feedback->priority == 'medium') selected @endif value="medium">
                                            @lang('app.medium')</option>
                                        <option @if ($feedback->priority == 'high') selected @endif value="high">
                                            @lang('app.high')</option>
                                        <option @if ($feedback->priority == 'urgent') selected @endif value="urgent">
                                            @lang('app.urgent')</option>
                                    </x-forms.select>
                                </div>
                                <div class="more-filter-items mb-4">
                                    <x-forms.label class="my-3" fieldId="ticket_type_id" :fieldLabel="__('modules.invoices.type')">
                                    </x-forms.label>
                                    <x-forms.input-group>
                                        <select class="form-control select-picker" name="type_id" id="ticket_type_id"
                                            data-container="body" data-live-search="true" data-size="8">
                                            <option value="">--</option>
                                            @foreach ($types as $type)
                                                <option @if ($type->id == $feedback->type_id) selected @endif
                                                    value="{{ $type->id }}">
                                                    {{ mb_ucwords($type->type) }}</option>
                                            @endforeach
                                        </select>
                                        @if ($viewFeedbackPermission == 'all')
                                            <x-slot name="append">
                                                <button id="addTicketType" type="button"
                                                    class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                            </x-slot>
                                        @endif
                                    </x-forms.input-group>
                                </div>
                                <div class="more-filter-items mb-4">
                                    <x-forms.label class="my-3" fieldId="ticket_channel_id" :fieldLabel="__('modules.tickets.channelName')">
                                    </x-forms.label>
                                    <x-forms.input-group>
                                        <select class="form-control select-picker" name="channel_id"
                                            id="ticket_channel_id" data-container="body" data-live-search="true"
                                            data-size="8">
                                            <option value="">--</option>
                                            @foreach ($channels as $channel)
                                                <option @if ($channel->id == $feedback->channel_id) selected @endif
                                                    value="{{ $channel->id }}">
                                                    {{ mb_ucwords($channel->channel_name) }}
                                                </option>
                                            @endforeach
                                        </select>
                                        @if ($viewFeedbackPermission == 'all')
                                            <x-slot name="append">
                                                <button id="addChannel" type="button"
                                                    class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                            </x-slot>
                                        @endif
                                    </x-forms.input-group>
                                </div>
                                <div class="more-filter-items">
                                    <x-forms.select fieldId="ticket-status" :fieldLabel="__('app.status')" fieldName="status"
                                        data-container="body">
                                        <option @if ($feedback->status == 'open') selected @endif value="open"
                                            data-content="<i class='fa fa-circle mr-2 text-red'></i>{{ __('app.open') }}">
                                            @lang('app.open')
                                        </option>
                                        <option @if ($feedback->status == 'pending') selected @endif value="pending"
                                            data-content="<i class='fa fa-circle mr-2 text-yellow'></i>{{ __('app.pending') }}">
                                            @lang('app.pending')</option>
                                        <option @if ($feedback->status == 'resolved') selected @endif value="resolved"
                                            data-content="<i class='fa fa-circle mr-2 text-dark-green'></i>{{ __('app.resolved') }}">
                                            @lang('app.resolved')</option>
                                        <option @if ($feedback->status == 'closed') selected @endif value="closed"
                                            data-content="<i class='fa fa-circle mr-2 text-blue'></i>{{ __('app.closed') }}">
                                            @lang('app.closed')</option>
                                    </x-forms.select>
                                </div>
                                <div class="more-filter-items">
                                    <x-forms.label class="my-3" fieldId="tags" :fieldLabel="__('modules.tickets.tags')">
                                    </x-forms.label>
                                    <input type="text" name="tags" id="tags" class="rounded f-14"
                                        value="{{ implode(',', $feedback->feedbackTags->pluck('tag_name')->toArray()) }}">
                                </div>
                            </div>
                            <!-- TICKET FILTERS END -->
                            <!-- TICKET UPDATE START -->
                            <div class="ticket-update bg-white px-4 py-3">
                                <x-forms.button-primary class="ml-none d-flex submit-ticket-2 fixed-bottom">
                                    @lang('app.update')
                                </x-forms.button-primary>
                            </div>
                            <!-- TICKET UPDATE END -->
                        </x-form>
                    </div>
                    <!-- DETAILS END -->
                    <!-- CONTACT START -->
                    <div class="tab-pane fade" id="nav-contact" role="tabpanel" aria-labelledby="nav-contact-tab">
                        <!-- CONTACT OWNER START  -->
                        <div class="card-horizontal bg-white-shade ticket-contact-owner p-4 rounded-0">
                            <div class="card-img mr-3">
                                <img class="___class_+?88___" src="{{ $feedback->requester->image_url }}"
                                    alt="{{ mb_ucwords($feedback->requester->name) }}">
                            </div>
                            <div class="card-body border-0 p-0 w-100">
                                <h4 class="card-title f-14 font-weight-normal mb-0">
                                    <a class="text-dark-grey"
                                        @if ($feedback->requester->hasRole('employee')) href="{{ route('employees.show', $feedback->requester->id) }}"
                                    @else
                                        href="{{ route('clients.show', $feedback->requester->id) }}" @endif>
                                        {{ mb_ucwords($feedback->requester->name) }}
                                    </a>
                                </h4>
                                @if ($feedback->requester->country_id)
                                    <span class="card-text f-12 text-dark-grey text-capitalize d-flex align-items-center">
                                        <span
                                            class='flag-icon flag-icon-{{ strtolower($feedback->requester->country->iso) }} mr-2'></span>
                                        {{ $feedback->requester->country->nicename }}
                                    </span>
                                @else
                                    --
                                @endif

                            </div>
                        </div>
                        <!-- CONTACT OWNER END  -->
                        <!-- TICKET CHART START  -->
                        <x-cards.data :title="__('app.menu.tickets')" padding="false">
                            <x-pie-chart id="ticket-chart" :labels="$feedbackChart['labels']" :values="$feedbackChart['values']" :colors="$feedbackChart['colors']"
                                height="200" width="220" />
                        </x-cards.data>
                        <!-- TICKET CHART END  -->
                        <!-- RECENT TICKETS START -->
                        <div class="card pt-4 px-4 border-grey border-left-0 border-right-0 rounded-0">
                            <div class="card-title">
                                <h4 class="f-18 f-w-500 text-capitalize mb-3">@lang('modules.tickets.recentTickets')</h4>
                            </div>
                            <!-- CHART START -->
                            <div class="card-body p-0">
                                <div class="recent-ticket position-relative" data-menu-vertical="1" data-menu-scroll="1"
                                    data-menu-dropdown-timeout="500" id="recentTickets">
                                    <div class="recent-ticket-inner position-relative">
                                        @foreach ($feedback->requester->tickets as $item)
                                            <div class="r-t-items d-flex">
                                                <div class="r-t-items-left text-lightest f-21">
                                                    <i class="fa fa-ticket-alt"></i>
                                                </div>
                                                <div class="r-t-items-right ">
                                                    <h3 class="f-14 font-weight-bold">
                                                        <a class="text-dark"
                                                            href="{{ route('feedback.show', $item->id) }}">{{ $item->subject }}</a>
                                                    </h3>
                                                    <span class="d-flex mb-1">
                                                        <span
                                                            class="mr-3 f-w-500 text-dark-grey">#{{ $item->id }}</span>
                                                        @if ($item->status == 'open')
                                                            @php
                                                                $statusColor = 'red';
                                                            @endphp
                                                        @elseif($item->status == 'pending')
                                                            @php
                                                                $statusColor = 'yellow';
                                                            @endphp
                                                        @elseif($item->status == 'resolved')
                                                            @php
                                                                $statusColor = 'dark-green';
                                                            @endphp
                                                        @elseif($item->status == 'closed')
                                                            @php
                                                                $statusColor = 'blue';
                                                            @endphp
                                                        @endif
                                                        <span class="f-13 text-darkest-grey text-capitalize">
                                                            <x-status :color="$statusColor" :value="$item->status" />
                                                        </span>

                                                    </span>
                                                    <p class="f-12 text-dark-grey">
                                                        {{ $item->created_at->timezone(company()->timezone)->translatedFormat(company()->date_format . ' ' . company()->time_format) }}
                                                    </p>
                                                </div>
                                            </div><!-- item end -->
                                        @endforeach

                                    </div>
                                </div>
                            </div>
                            <!-- CHART END -->
                        </div>
                        <!-- RECENT TICKETS END -->
                    </div>
                    <!-- CONTACT END -->

                    <div class="tab-pane fade" id="nav-other" role="tabpanel" aria-labelledby="nav-other-tab">
                        <x-form id="updateOther">
                            <!-- TICKET FILTERS START -->
                            <div class="ticket-filters p-4 w-100 position-relative border-bottom">
                                <x-forms.custom-field-show :fields="$fields"
                                    :model="$feedback"></x-forms.custom-field-show>
                            </div>
                            <!-- TICKET FILTERS END -->
                        </x-form>
                    </div>
                </div>
            </div>
            <!-- TICKET RIGHT END -->
        @endif
    </div>
    <!-- TICKET END -->
@endsection

@push('scripts')
    <script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
    <script src="{{ asset('vendor/jquery/tagify.min.js') }}"></script>

    <script>
        $(document).ready(function() {
            quillImageLoad('#description');
        });

        $('.reply-button').click(function() {
            $('#reply-section-action').toggleClass('d-none d-flex');
            $('#reply-section-action-2').toggleClass('d-none flex-row');
            $('#reply-section').removeClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        $('#cancel-reply').click(function() {
            $('#reply-section-action').toggleClass('d-none d-flex');
            $('#reply-section-action-2').toggleClass('d-none flex-row');
            $('#reply-section').addClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        $('#add-file').click(function() {
            $('.upload-section').removeClass('d-none');
            $('#add-file').addClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        var input = document.querySelector('input[name=tags]'),
            // init Tagify script on the above inputs
            tagify = new Tagify(input);

        Dropzone.autoDiscover = false;
        //Dropzone class
        taskDropzone = new Dropzone("div#task-file-upload-dropzone", {
            dictDefaultMessage: "{{ __('app.dragDrop') }}",
            url: "{{ route('feedback-files.store') }}",
            headers: {
                'X-CSRF-TOKEN': '{{ csrf_token() }}'
            },
            paramName: "file",
            maxFilesize: DROPZONE_MAX_FILESIZE,
            maxFiles: 10,
            autoProcessQueue: false,
            uploadMultiple: true,
            addRemoveLinks: true,
            parallelUploads: 10,
            acceptedFiles: DROPZONE_FILE_ALLOW,
            init: function() {
                taskDropzone = this;
            }
        });
        taskDropzone.on('sending', function(file, xhr, formData) {
            var ids = $('#feedback_reply_id').val();
            formData.append('feedback_reply_id', ids);
            formData.append('feedback_id', '{{ $feedback->id }}');
            $.easyBlockUI();
        });
        taskDropzone.on('uploadprogress', function() {
            $.easyBlockUI();
        });
        taskDropzone.on('completemultiple', function() {
            var msgs = "@lang('messages.addDiscussion')";
            window.location.href = "{{ route('feedback.show', $feedback->feedback_number) }}";
        });

        $('.submit-ticket').click(function() {
            var note = document.getElementById('description').children[0].innerHTML;
            document.getElementById('description-text').value = note;

            var status = $(this).data('status');
            $('#status').val(status);

            const url = "{{ route('feedback.update', $feedback->id) }}";
            var roleInput = document.getElementById('role').value;

            $.easyAjax({
                url: url,
                container: '#ticketMsg',
                type: "POST",
                blockUI: true,
                data: $('#updateTicket2').serialize(),
                success: function(response) {

                    if (response.status == 'success') {
                        if (taskDropzone.getQueuedFiles().length > 0) {
                            $('#feedback_reply_id').val(response.reply_id);
                            taskDropzone.processQueue();
                        } else {
                            window.location.href =
                                "{{ route('feedback.show', $feedback->feedback_number) }}";
                        }
                    }
                }
            });
        });

        $('.submit-ticket-2').click(function() {

            $.easyAjax({
                url: "{{ route('feedback.update_other_data', $feedback->id) }}",
                container: '#updateTicket1',
                type: "POST",
                blockUI: true,
                disableButton: true,
                buttonSelector: ".submit-ticket-2",
                data: $('#updateTicket1').serialize(),
                success: function(response) {
                    if (response.status == 'success') {
                        var status = $('#ticket-status').val();

                        ($('#ticket-status').val() != 'closed') ? $('#ticket-closed').show(): $(
                            '#ticket-closed').hide();

                        switch (status) {
                            case 'open':
                                var statusHtml =
                                    '<i class="fa fa-circle mr-2 text-red"></i>@lang('app.open')';
                                break;
                            case 'pending':
                                var statusHtml =
                                    '<i class="fa fa-circle mr-2 text-yellow"></i>@lang('app.pending')';
                                break;
                            case 'resolved':
                                var statusHtml =
                                    '<i class="fa fa-circle mr-2 text-dark-green"></i>@lang('app.resolved')';
                                break;
                            case 'closed':
                                var statusHtml =
                                    '<i class="fa fa-circle mr-2 text-blue"></i>@lang('app.closed')';
                                break;

                            default:
                                var statusHtml =
                                    '<i class="fa fa-circle mr-2 text-red"></i>@lang('app.open')';
                                break;
                        }
                        $('.ticket-status').html(statusHtml);
                    }
                }
            })
        });

        $('.apply-template').click(function() {
            var templateId = $(this).data('template-id');

            $.easyAjax({
                url: "{{ route('feedbackTemplates.fetchTemplate') }}",
                data: {
                    templateId: templateId
                },
                success: function(response) {
                    if (response.status == "success") {
                        var container = $('#description').get(0);
                        var quill = new Quill(container);
                        quill.clipboard.dangerouslyPasteHTML(0, response.replyText);
                    }
                }
            })
        })

        $('body').on('click', '.delete-file', function() {
            var id = $(this).data('row-id');
            var replyFile = $(this);
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedback-files.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                replyFile.closest('.card').remove();
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.delete-message', function() {
            var id = $(this).data('row-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedback-replies.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('#message-' + id).remove();
                            }
                        }
                    });
                }
            });
        });

        function scrollToBottom(divId) {
            var myDiv = document.getElementById(divId);
            myDiv.scrollTop = myDiv.scrollHeight;
        }

        scrollToBottom('ticketMsg');

        /* delete agent */
        $('body').on('click', '.delete-agents', function() {
            var id = $(this).data('agent-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeAgentText')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedback-agents.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        /* change agent status */
        $('body').on('change', '.change-agent-status', function() {

            var agentId = $(this).data('agent-id');
            var status = $(this).val();

            var token = '{{ csrf_token() }}';
            var url = "{{ route('feedback-agents.update', ':id') }}";
            url = url.replace(':id', agentId);

            if (typeof agentId !== 'undefined') {
                $.easyAjax({
                    type: 'PUT',
                    url: url,
                    blockUI: true,
                    data: {
                        '_token': token,
                        'status': status
                    }
                });
            }
        });


        $('body').on('click', '.delete-type', function() {
            var id = $(this).data('type-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeTicketText')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedbackTypes.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.edit-type', function() {
            var typeId = $(this).data('type-id');
            var url = "{{ route('feedbackTypes.edit', ':id') }}";
            url = url.replace(':id', typeId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('body').on('click', '.delete-channel', function() {
            var id = $(this).data('channel-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeChannelText')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedbackChannels.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.edit-channel', function() {
            var typeId = $(this).data('channel-id');
            var url = "{{ route('feedbackChannels.edit', ':id') }}";
            url = url.replace(':id', typeId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open edit reply template modal */
        $('body').on('click', '.edit-template', function() {
            var templateId = $(this).data('template-id');
            var url = "{{ route('feedbackTemplates.edit', ':id') }}";
            url = url.replace(':id', templateId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* delete template */
        $('body').on('click', '.delete-template', function() {
            var id = $(this).data('template-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeTemplateText')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('feedbackTemplates.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        /* open add agent modal */
        $('body').on('click', '#addAgent', function() {
            var url = "{{ route('feedback-agents.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open add agent modal */
        $('body').on('click', '#addChannel', function() {
            var url = "{{ route('feedbackChannels.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open creat reply template modal */
        $('body').on('click', '#addReplyTemplate', function() {
            var url = "{{ route('feedbackTemplates.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open add agent modal */
        $('body').on('click', '#addTicketType', function() {
            var url = "{{ route('feedbackTypes.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* change agent group */
        $('.change-agent-group').change(function() {

            var agentId = $(this).data('agent-id');
            var groupId = $(this).val();
            var token = '{{ csrf_token() }}';
            var url = "{{ route('feedback_agents.update_group', ':id') }}";
            url = url.replace(':id', agentId);

            $.easyAjax({
                type: 'POST',
                url: url,
                blockUI: true,
                data: {
                    '_token': token,
                    'groupId': groupId
                }
            });
        });
    </script>
@endpush
