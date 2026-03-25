<div class="modal-body wrapper">
    <div class="row">
        <div class="col-md-12 col-lg-12">
            <div class="detail-group">
                <b>{{__('Request Detail')}}</b>
                <p class="mb-20">{{ $wORequest->request_detail }} </p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Client')}}</b>
                <p class="mb-20">{{ !empty($wORequest->clients)?$wORequest->clients->name:'-' }}</p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Asset')}}</b>
                <p class="mb-20">{{ !empty($wORequest->assets)?$wORequest->assets->name:'-' }}</p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Due Date')}}</b>
                <p class="mb-20">{{ dateFormat($wORequest->due_date) }}  </p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Priority')}}</b>
                <p class="mb-20">
                    @if($wORequest->priority=='low')
                        <span
                            class="badge text-bg-primary">{{\App\Models\WORequest::$priority[$wORequest->priority] }}</span>
                    @elseif($wORequest->priority=='medium')
                        <span
                            class="badge text-bg-info">{{\App\Models\WORequest::$priority[$wORequest->priority] }}</span>
                    @elseif($wORequest->priority=='high')
                        <span
                            class="badge text-bg-warning">{{\App\Models\WORequest::$priority[$wORequest->priority] }}</span>
                    @elseif($wORequest->priority=='critical')
                        <span
                            class="badge text-bg-danger">{{\App\Models\WORequest::$priority[$wORequest->priority] }}</span>
                    @endif
                </p>
            </div>
        </div>

        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Status')}}</b>
                <p class="mb-20">
                    @if($wORequest->status=='pending')
                        <span
                            class="badge text-bg-warning">{{\App\Models\WORequest::$status[$wORequest->status] }}</span>
                    @elseif($wORequest->status=='in_progress')
                        <span
                            class="badge text-bg-primary">{{\App\Models\WORequest::$status[$wORequest->status] }}</span>
                    @elseif($wORequest->status=='completed')
                        <span
                            class="badge text-bg-success">{{\App\Models\WORequest::$status[$wORequest->status] }}</span>
                    @elseif($wORequest->status=='cancel')
                        <span
                            class="badge text-bg-danger">{{\App\Models\WORequest::$status[$wORequest->status] }}</span>
                    @endif
                </p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Assign')}}</b>
                <p class="mb-20"> {{!empty($wORequest->assigned)?$wORequest->assigned->name:'-' }}</p>
            </div>
        </div>
        <div class="col-md-12 col-lg-12">
            <div class="detail-group">
                <b>{{__('Notes')}}</b>
                <p class="mb-20">{{ !empty($wORequest->notes)?$wORequest->notes:"-" }}</p>
            </div>
        </div>
        <hr>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Preferred Date')}}</b>
                <p class="mb-20"> {{ dateFormat($wORequest->preferred_date) }}</p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Preferred Time')}}</b>
                <p class="mb-20"> {{$wORequest->preferred_time}}</p>
            </div>
        </div>

        <div class="col-md-12 col-lg-12">
            <div class="detail-group">
                <b>{{__('Preference Note')}}</b>
                <p class="mb-20">{{ !empty($wORequest->preferred_note)?$wORequest->preferred_note:"-" }}</p>
            </div>
        </div>
    </div>

</div>

