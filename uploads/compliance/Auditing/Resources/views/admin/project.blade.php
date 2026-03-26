@extends('layouts.app')

@section('page-title')
    <div class="row bg-title">
        <div class="col-lg-8 col-md-5 col-sm-6 col-xs-12">
            <h4 class="page-title"><i class="{{ $pageIcon ?? 'fa fa-history' }}"></i> {{ $pageTitle ?? __('Audit Log') }}
            </h4>
        </div>
    </div>
@endsection

@push('head-script')
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.13/css/dataTables.bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.1.1/css/responsive.bootstrap.min.css">
    <link rel="stylesheet" href="//cdn.datatables.net/buttons/1.2.2/css/buttons.dataTables.min.css">
    <link rel="stylesheet" href="//cdn.datatables.net/buttons/1.2.2/css/buttons.dataTables.min.css">
    <link rel="stylesheet" href="{{ asset('plugins/bower_components/bootstrap-select/bootstrap-select.min.css') }}">
    <link rel="stylesheet" href="{{ asset('plugins/bower_components/custom-select/custom-select.css') }}">
@endpush

@section('content')
    <div class="row">

        <div class="col-md-12">
            <div class="white-box">
                @section('filter-section')
                <form>
                    <div class="form-group">
                        <select name="year" id="year" class="form-control">
                            @foreach (range(date("Y"), 2015) as $year)
                                <option value="{{ $year }}"
                                    {{ request('year') == $year ? 'selected' : '' }}
                                    {{ request('year') == null && $year == date('Y') ? 'selected' : '' }}>
                                    {{ $year }}
                                </option>
                            @endforeach
                        </select>
                    </div>

                    <div class="form-group">
                        <select name="month" id="month" class="form-control">
                            @for($i = 1 ; $i <= 12; $i++)
                                <option value="{{ $i }}"
                                    {{ request('month') == $i ? 'selected' : '' }}
                                    {{ request('month') == null && $i == date('m') ? 'selected' : '' }}>
                                    {{ date("F",strtotime((request()->year ?? date("Y"))."-".$i."-01")) }}
                                </option>
                            @endfor
                        </select>
                    </div>

                    <div class="col-md-12">
                        <div class="form-group">
                            <button class="btn btn-success btn-sm col-md-5">
                                <i class="fa fa-check"></i>
                                @lang('app.apply')
                            </button>
                            <a href="?" class="btn btn-inverse col-md-5 btn-sm col-md-offset-1">
                                <i class="fa fa-refresh"></i>
                                @lang('app.reset')
                            </a>
                        </div>
                    </div>
                </form>
                @endsection
                
                <div class="table-responsive">
                    {!! $dataTable->table(['class' => 'table table-bordered table-hover toggle-circle default footable-loaded footable']) !!}
                </div>
            </div>
        </div>
    </div>
    <!-- .row -->
@endsection

@push('footer-script')
<script src="{{ asset('plugins/bower_components/datatables/jquery.dataTables.min.js') }}"></script>
<script src="https://cdn.datatables.net/1.10.13/js/dataTables.bootstrap.min.js"></script>
<script src="https://cdn.datatables.net/buttons/1.0.3/js/dataTables.buttons.min.js"></script>
<script src="{{ asset('js/datatables/buttons.server-side.js') }}"></script>
{!! $dataTable->scripts() !!}
<script>
    $('#task_log').on('click', '.show-task-detail', function () {
            $(".right-sidebar").slideDown(50).addClass("shw-rside");

            var id = $(this).data('task-id');
            var url = "{{ route('admin.all-tasks.show',':id') }}";
            url = url.replace(':id', id);

            $.easyAjax({
                type: 'GET',
                url: url,
                success: function (response) {
                    if (response.status == "success") {
                        $('#right-sidebar-content').html(response.view);
                    }
                }
            });
        })
</script>
@endpush
