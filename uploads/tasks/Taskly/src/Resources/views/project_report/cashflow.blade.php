<div class="row justify-content-between mt-3 mb-3">
    <div class="col-xl-3">
        <h4 class="m-b-10">{{__('Cashflow')}}
        </h4>
    </div>
    <div class="col-xl-9">
        <div class="float-end">
            <a href="#" class="btn btn-sm btn-primary" onclick="saveAsPDF()"data-bs-toggle="tooltip" title="{{__('Download')}}" data-original-title="{{__('Download')}}">
                <span class="btn-inner--icon"><i class="ti ti-download"></i></span>
            </a>
        </div>
    </div>
</div>
<div class="row justify-content-center">
    <div class="col-sm-12">
            <div class="card">
                <div class="card-body">
                    <div class="col-xl-12">
                        <div class="row justify-content-between">
                            <div class="col-xl-3">
                                <ul class="nav nav-pills my-3" id="pills-tab" role="tablist">
                                    <li class="nav-item">
                                        <a class="nav-link" id="pills-home-tab" data-bs-toggle="pill"
                                           href="{{ route('project_report.show',$id) }}"
                                           onclick="window.location.href = '{{ route('project_report.show',$id) }}'" role="tab"
                                           aria-controls="pills-home" aria-selected="true">{{ __('Monthly') }}</a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link active" id="pills-profile-tab" data-bs-toggle="pill" href="#" role="tab"
                                           aria-controls="pills-profile" aria-selected="false">{{ __('Quarterly') }}</a>
                                    </li>
                                </ul>
                            </div>
                            <div class="col-xl-9">
                                <div class="row justify-content-end align-items-center">
                                    <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                        <div class="btn-box">
                                            {{ Form::label('year', __('Year'),['class'=>'form-label '])}}
                                            {{ Form::select('year',$data['yearList'],isset($year) ? $year : '', array('class' => 'form-control select year')) }}
                                        </div>
                                    </div>

                                    <div class="col-auto d-flex mt-4">
                                        <a  class="btn btn-sm btn-primary search me-2" data-bs-toggle="tooltip" title="{{__('Apply')}}" data-original-title="{{__('apply')}}">
                                            <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                                        </a>
                                        <a href="{{route('project_report.show',$id)}}" class="btn btn-sm btn-danger " data-bs-toggle="tooltip"  title="{{ __('Reset') }}" data-original-title="{{__('Reset')}}">
                                            <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off "></i></span>
                                        </a>

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
    </div>
</div>

<div id="printableArea3">
    <div class="row mt-1">
        <div class="col">
            <input type="hidden" value="{{__('Quarterly Cashflow').' '.'Report of'.' '.$filter['startDateRange'].' to '.$filter['endDateRange']}}" id="filenames">
            <div class="card p-4 mb-4">
                <h7 class="report-text gray-text mb-0">{{__('Report')}} :</h7>
                <h6 class="report-text mb-0">{{__('Quarterly Cashflow')}}</h6>
            </div>
        </div>
        <div class="col">
            <div class="card p-4 mb-4">
                <h7 class="report-text gray-text mb-0">{{__('Duration')}} :</h7>
                <h6 class="report-text mb-0">{{$filter['startDateRange'].' to '.$filter['endDateRange']}}</h6>
            </div>
        </div>
    </div>

    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-body table-border-style">
                    <div class="row">
                        <div class="col-sm-12">
                            <h5 class="pb-3">{{__('Income')}}</h5>
                            <div class="table-responsive mt-3 mb-3">
                                <table class="table">
                                    <thead>
                                    <tr>
                                        <th width="25%">{{__('Category')}}</th>
                                        @foreach($data['month'] as $m)
                                            <th width="15%">{{$m}}</th>
                                        @endforeach
                                    </tr>
                                    </thead>
                                    <tbody>

                                    <div class="row">
                                        <div class="col-sm-12">
                                            <table class="table table-flush border">
                                                <tbody>
                                                <tr>
                                                    <td width="25%" class="text-dark">{{__('Total Income (Invoice)')}}</td>
                                                    @foreach($data['totalIncome'] as $income)
                                                        <td width="15%">{{currency_format_with_sym($income)}}</td>
                                                    @endforeach
                                                </tr>

                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    </tbody>
                                </table>
                            </div>

                            <div class="col-sm-12">
                                <h5>{{__('Expense')}}</h5>
                                <div class="table-responsive mt-4">
                                    <table class="table mb-0" >
                                        <thead>
                                        <tr>
                                            <th width="25%">{{__('Category')}}</th>
                                                @foreach($data['month'] as $m)
                                                    <th width="15%">{{$m}}</th>
                                                @endforeach
                                        </tr>
                                        </thead>
                                        <tbody>
                                        <div class="row">
                                            <div class="col-sm-12">
                                                <table class="table table-flush border" >
                                                    <tbody>
                                                    <tr>
                                                        <td class="text-dark">{{__('Total Expenses (Bill)')}}</td>
                                                        @foreach($data['totalExpense'] as $income)
                                                            <td width="15%">{{currency_format_with_sym($income)}}</td>
                                                        @endforeach
                                                    </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>

                                        </tbody>
                                    </table>
                                </div>
                                <div class="row">
                                    <div class="col-sm-12">
                                        <table class="table table-flush border" >
                                            <tbody>
                                                <thead>
                                                <tr>
                                                    <th colspan="13" class="font-bold"><span>{{__('Net Profit = Total Income - Total Expense ')}}</span></th>
                                                </tr>
                                                </thead>
                                                <tr>
                                                    <td width="25%" class="text-dark">{{__('Net Profit')}}</td>
                                                    @foreach($data['netProfitArray'] as $i=>$profit)
                                                        <td width="15%"> {{currency_format_with_sym($profit)}}</td>
                                                    @endforeach
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
$(document).on('click', '.search', function(event) {
    event.preventDefault();
    var data = {
          year : $('.year').val(),
    };
    $.ajax({
        url: '{{route('projectreport.quarterly.cashflow',$id)}}',
        type: 'POST',
        data: data,
        success: function (data) {
            $('.quarterly_cashflow').html(data.html);
        },
    })
});
</script>

<script>
    var filename = $('#filenames').val();

    function saveAsPDF() {
        var element = document.getElementById('printableArea3');
        var opt = {
            margin: 0.3,
            filename: filename,
            image: {type: 'jpeg', quality: 1},
            html2canvas: {scale: 4, dpi: 72, letterRendering: true},
            jsPDF: {unit: 'in', format: 'A2'}
        };
        html2pdf().set(opt).from(element).save();
    }
</script>
