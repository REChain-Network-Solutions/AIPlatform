<div class="col-md-12 col-sm-12">
    <div class="panel-group wrap mb-0" id="bs-collapse">
        @foreach ($properties as $key => $property)
            <div class="panel">
                <div class="panel-heading">
                    <h6 class="panel-title">
                        <a data-toggle="collapse" data-parent="#"
                            href="#{{ $id }}-col-{{ $loop->iteration }}">
                            {{ str_replace('_', ' ', $key) }}
                        </a>
                    </h6>
                </div>
                <div id="{{ $id }}-col-{{ $loop->iteration }}" class="panel-collapse collapse">
                    @if (is_array($property))
                        <div class="panel-body">
                            <div>
                                <h6 style="font-weight: bold">@lang('auditlog::app.previous_value')</h6>
                                <p>{{ in_array($key, ['leave_date']) ? \Carbon\Carbon::create($property['original'])->format($global->date_format) : $property['original'] }}
                                </p>
                            </div>
                            <hr>
                            <div>
                                <h6 style="font-weight: bold">@lang('auditlog::app.present_value')</h6>
                                <p>{{ in_array($key, ['leave_date']) ? \Carbon\Carbon::create($property['original'])->format($global->date_format) : $property['changes'] }}
                                </p>
                            </div>
                        </div>
                    @endif
                </div>
            </div>
        @endforeach
    </div>
</div>
