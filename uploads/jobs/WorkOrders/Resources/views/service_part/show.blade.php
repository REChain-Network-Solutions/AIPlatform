<div class="modal-body wrapper">
    <div class="row">
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Title')}}</b>
                <p class="mb-20">{{ $servicePart->title }} </p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('SKU')}}</b>
                <p class="mb-20">{{ $servicePart->sku }} </p>
            </div>
        </div>

        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Price')}}</b>
                <p class="mb-20"> {{ priceFormat($servicePart->price) }}</p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Unit')}}</b>
                <p class="mb-20">{{ $servicePart->unit }}</p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Type')}}</b>
                <p class="mb-20">{{ ucfirst($servicePart->type) }} </p>
            </div>
        </div>
        <div class="col-md-6 col-lg-6">
            <div class="detail-group">
                <b>{{__('Description')}}</b>
                <p class="mb-20">{{ !empty($servicePart->description)?$servicePart->description:"-" }}</p>
            </div>
        </div>
    </div>

    @if(count($servicePart->serviceTasks)>0)
    <div class=" col-md-12 mb-20">
        <h5> {{__('Service Tasks')}}</h5>
    </div>

    <div class="row">
        <div class="table-responsive">
            <table class="table">
                <thead>
                <tr>
                    <th>{{__('Task')}}</th>
                    <th>{{__('Duration')}}</th>
                    <th>{{__('Description')}}</th>
                </tr>
                </thead>
                <tbody>
                @foreach($servicePart->serviceTasks as $task)
                    <tr>
                        <td>{{$task->task}}</td>
                        <td>{{$task->duration}}</td>
                        <td>{{$task->description}}</td>
                    </tr>
                @endforeach
                </tbody>
            </table>
        </div>
    </div>
    @endif
</div>

