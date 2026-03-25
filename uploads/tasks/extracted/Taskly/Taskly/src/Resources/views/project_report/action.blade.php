@permission('task manage')
<div class="action-btn">
    <a  class=" btn btn-sm align-items-center text-white bg-warning" data-toggle="tooltip"  title="{{__('view')}}" data-size="lg" data-title="{{__('view')}}" href="{{route('project_report.show', [$project->id])}}"><i class="ti ti-eye"></i></a>
</div>
@endpermission
