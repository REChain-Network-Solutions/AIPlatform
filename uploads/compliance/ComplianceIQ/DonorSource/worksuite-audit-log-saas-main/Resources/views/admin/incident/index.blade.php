@extends('layouts.app')

@section('page-title')
    <div class="row bg-title">
        <div class="col-lg-8 col-md-5 col-sm-6 col-xs-12">
            <h4 class="page-title"><i class="{{ $pageIcon ?? 'fa fa-history' }}"></i>
                {{ $pageTitle ?? __('Audit Log') }}
            </h4>
        </div>
    </div>
    <style>
        .wrap {
            box-shadow: 0px 2px 2px 0px rgba(0, 0, 0, 0.14), 0px 3px 1px -2px rgba(0, 0, 0, 0.2), 0px 1px 5px 0px rgba(0, 0, 0, 0.12);
            border-radius: 4px;
        }


        .panel {
            border-width: 0 0 1px 0;
            border-style: solid;
            border-color: #fff;
            background: none;
            box-shadow: none;
        }

        .panel:last-child {
            border-bottom: none;
        }

        .panel-heading {
            background-color: #009688;
            border-radius: 0;
            border: none;
            color: #fff;
            padding: 0;
        }

        .panel-title a {
            display: block;
            color: #fff;
            position: relative;
            font-size: 12px;
            font-weight: 400;
            text-transform: capitalize
        }

        .panel-body {
            background: #fff;
        }

        .panel-group {
            margin-bottom: 0px
        }

        .panel:last-child .panel-heading {
            border-radius: 0 0 4px 4px;
            transition: border-radius 0.3s linear 0.2s;

        }

        .panel:last-child .panel-heading.active {
            border-radius: 0;
            transition: border-radius linear 0s;
        }

        /* #bs-collapse icon scale option */

        .panel-heading a:before {
            content: '\e146';
            position: absolute;
            font-family: 'Material Icons';
            right: 5px;
            top: -4px;
            font-size: 24px;
            transition: all 0.5s;
            transform: scale(1);
        }

        .panel-heading.active a:before {
            content: ' ';
            transition: all 0.5s;
            transform: scale(0);
        }

        #bs-collapse .panel-heading a:after {
            content: ' ';
            font-size: 24px;
            position: absolute;
            font-family: 'Material Icons';
            right: 5px;
            top: 10px;
            transform: scale(0);
            transition: all 0.5s;
        }

        #bs-collapse .panel-heading.active a:after {
            content: '\e909';
            transform: scale(1);
            transition: all 0.5s;
        }

        /* #accordion rotate icon option */

        #accordion .panel-heading a:before {
            content: '\e316';
            font-size: 24px;
            position: absolute;
            font-family: 'Material Icons';
            right: 5px;
            top: 10px;
            transform: rotate(180deg);
            transition: all 0.5s;
        }

        #accordion .panel-heading.active a:before {
            transform: rotate(0deg);
            transition: all 0.5s;
        }

    </style>
@endsection

@push('head-script')

    <link rel="stylesheet" href="{{ asset('plugins/bower_components/custom-select/custom-select.css') }}">
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.css" />

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
                    <div class="form-group col-md-12">
                        <label class="control-label required" for="date">@lang('auditlog::app.dateRange')</label>
                        <input type="text" name="daterange" class="form-control" autocomplete="off"
                            value="{{ request('daterange') ??
                                now()->subMonth()->format($global->date_format) .
                                    ' - ' .
                                    now()->format($global->date_format) }}">
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
                {{-- <div class="col-md-12">
                  <div class="form-group p-t-10">
                      <a href="{{ route('admin.audit-log.attendance-export',['daterange' => request('daterange')]) }}" class="btn btn-inverse col-md-5 btn-sm">
                          <i class="ti-export" style="padding-right: 5px"></i>
                          @lang('auditlog::app.export')
                      </a>
                  </div>
                </div> --}}
            @endsection

            <div class="table-responsive">
                {!! $dataTable->table(['class' => 'table table-bordered table-hover toggle-circle default footable-loaded footable']) !!}
            </div>
        </div>
    </div>
</div>
{{-- Ajax Modal --}}
<div class="modal fade bs-modal-md in" id="incidentModal" role="dialog" aria-labelledby="myModalLabel"
    aria-hidden="true">
    <div class="modal-dialog modal-md" id="modal-data-application">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal" aria-hidden="true"></button>
                <span class="caption-subject font-red-sunglo bold uppercase" id="modelHeading"></span>
            </div>
            <div class="modal-body">
                @lang('incident::app.loading')
            </div>
            <div class="modal-footer">
                <button type="button" class="btn default" data-dismiss="modal">Close</button>
                <button type="button" class="btn blue">Save changes</button>
            </div>
        </div>
        <!-- /.modal-content -->
    </div>
    <!-- /.modal-dialog -->.
</div>
{{-- Ajax Modal Ends --}}
<!-- .row -->
@endsection

@push('footer-script')
<script src="{{ asset('plugins/bower_components/datatables/jquery.dataTables.min.js') }}"></script>
<script src="https://cdn.datatables.net/1.10.13/js/dataTables.bootstrap.min.js"></script>
<script src="https://cdn.datatables.net/buttons/1.0.3/js/dataTables.buttons.min.js"></script>
<script src="{{ asset('js/datatables/buttons.server-side.js') }}"></script>

<script src="{{ asset('plugins/bower_components/custom-select/custom-select.min.js') }}"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/momentjs/latest/moment.min.js"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.min.js"></script>

{!! $dataTable->scripts() !!}

<script>
    $(document).ready(function() {

        $(document.body).on('click', '.view-assigned-incident', function(ev) {
            ev.preventDefault();
            let url = $(this).attr('href');
            $.ajaxModal('#incidentModal', url);
        });

        @if (session()->has('error_date'))
            toastr.error('{{ session()->get('error_date') }}');
        @endif

        $('input[name="daterange"]').daterangepicker({
            opens: 'left',
            locale: {
                format: '{{ daterangeFormat($global->date_picker_format) }}',
                daysOfWeek: [{!! arrayFormatJs(__('app.dayNames')) !!}],
                monthNames: [{!! arrayFormatJs(__('app.monthNames')) !!}],
            }
        });

        $(".select2").select2({
            formatNoMatches: function() {
                return "{{ __('messages.noRecordFound') }}";
            }
        });

        $('.collapse.in').prev('.panel-heading').addClass('active');
        $('#accordion, #bs-collapse')
            .on('show.bs.collapse', function(a) {
                $(a.target).prev('.panel-heading').addClass('active');
            })
            .on('hide.bs.collapse', function(a) {
                $(a.target).prev('.panel-heading').removeClass('active');
            });
    });
</script>
@endpush
