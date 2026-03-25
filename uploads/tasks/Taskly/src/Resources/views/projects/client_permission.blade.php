<form method="post" action="{{ route('projects.client.permission.store',[$currentWorkspace->slug,$project->id,$client->id]) }}">
    @csrf
    @include('projects.project_permission')
    <div class=" modal-footer">
       <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
<input type="submit" value="{{ __('Save Changes')}}" class="btn btn-primary">
    </div>
</form>
