
<div class="card sticky-top" style="top:30px">
    <div class="list-group list-group-flush" id="useradd-sidenav">
        <a href="{{route('projects.proposal',$id)}}" class="list-group-item list-group-item-action border-0 {{ (request()->is('*proposal') ? 'active' : '')}}">{{__('Proposals')}} <div class="float-end"><i class="ti ti-chevron-right"></i></div></a>
        @stack('project_retainer_tab')
        <a href="{{route('projects.invoice',$id)}}" class="list-group-item list-group-item-action border-0 {{ (request()->is('*invoice') ? 'active' : '')}}">{{__('Invoice')}} <div class="float-end"><i class="ti ti-chevron-right"></i></div></a>
        @stack('project_bill_tab')

    </div>
</div>
