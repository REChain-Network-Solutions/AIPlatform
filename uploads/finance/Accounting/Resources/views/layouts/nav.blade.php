<section class="no-print">
    <nav class="navbar-default tw-transition-all tw-duration-5000 tw-shrink-0 tw-rounded-2xl tw-m-[16px] tw-border-2 !tw-bg-white">
        <div class="container-fluid">
            <!-- Brand and toggle get grouped for better mobile display -->
            <div class="navbar-header">
                <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false" style="margin-top: 3px; margin-right: 3px;">
                    <span class="sr-only">Toggle navigation</span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                </button>
                <a class="navbar-brand" href="{{action([\Modules\Accounting\Http\Controllers\AccountingController::class, 'dashboard'])}}"><i class="fas fa fa-broadcast-tower"></i> {{__('accounting::lang.accounting')}}</a>
            </div>

            <!-- Collect the nav links, forms, and other content for toggling -->
            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <ul class="nav navbar-nav">
                    @if(auth()->user()->can('accounting.manage_accounts'))
                        <li @if(request()->segment(2) == 'chart-of-accounts') class="active" @endif><a href="{{action([\Modules\Accounting\Http\Controllers\CoaController::class, 'index'])}}">@lang('accounting::lang.chart_of_accounts')</a></li>
                    @endif
                    
                    @if(auth()->user()->can('accounting.view_journal'))
                        <li @if(request()->segment(2) == 'journal-entry') class="active" @endif><a href="{{action([\Modules\Accounting\Http\Controllers\JournalEntryController::class, 'index'])}}">@lang('accounting::lang.journal_entry')</a></li>
                    @endif

                    @if(auth()->user()->can('accounting.view_transfer'))
                        <li @if(request()->segment(2) == 'transfer') class="active" @endif>
                            <a href="{{action([\Modules\Accounting\Http\Controllers\TransferController::class, 'index'])}}">
                                @lang('accounting::lang.transfer')
                            </a>
                        </li>
                    @endif

                    <li @if(request()->segment(2) == 'transactions') class="active" @endif><a href="{{action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'index'])}}">@lang('accounting::lang.transactions')</a></li>

                    @if(auth()->user()->can('accounting.manage_budget'))
                        <li @if(request()->segment(2) == 'budget') class="active" @endif>
                            <a href="{{action([\Modules\Accounting\Http\Controllers\BudgetController::class, 'index'])}}">
                                @lang('accounting::lang.budget')
                            </a>
                        </li>
                    @endif
                    @if(auth()->user()->can('accounting.view_reports'))
                    <li @if(request()->segment(2) == 'reports') class="active" @endif><a href="{{action([\Modules\Accounting\Http\Controllers\ReportController::class, 'index'])}}">
                        @lang('accounting::lang.reports')
                    </a></li>
                    @endif

                    <li @if(request()->segment(2) == 'settings') class="active" @endif><a href="{{action([\Modules\Accounting\Http\Controllers\SettingsController::class, 'index'])}}">@lang('messages.settings')</a></li>
                </ul>

            </div><!-- /.navbar-collapse -->
        </div><!-- /.container-fluid -->
    </nav>
</section>