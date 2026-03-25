
<option value="Taskly" @if($type == 'project') selected @endif {{(isset($_GET['account_type']) && $_GET['account_type'] == 'Taskly') ? 'selected' : '' }}>
    {{ __('Projects') }}
</option>
