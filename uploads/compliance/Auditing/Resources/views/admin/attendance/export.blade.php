<table>
    <thead>
    <tr>
        <th>{{  __('auditlog::app._log_activity.id') }}</th>
        <th>{{  __('auditlog::app._log_activity.user') }}</th>
        <th>{{  __('auditlog::app._log_activity.for_whom') }}</th>
        <th>{{  __('auditlog::app._log_activity.activity') }}</th>
        <th>{{  __('auditlog::app._log_activity.properties') }}</th>
        <th>{{  __('auditlog::app._log_activity.ip') }}</th>
        <th>{{  __('auditlog::app._log_activity.date') }}</th>
    </tr>
    </thead>
    <tbody>
    @foreach($logActivities as $logActivity)
        <tr>
            <td>{{ $logActivity->id }}</td>
            <td>{{ $logActivity->name }}</td>
            <td>{{ $logActivity->whom_user_name }}</td>
            <td>{{ $logActivity->description }}</td>
            <td>{{ $logActivity->properties }}</td>
            <td>{{ $logActivity->ip }}</td>
            <td>{{ $logActivity->created_at->format('d M Y - h:i a') }}</td>
        </tr>
    @endforeach
    </tbody>
</table>