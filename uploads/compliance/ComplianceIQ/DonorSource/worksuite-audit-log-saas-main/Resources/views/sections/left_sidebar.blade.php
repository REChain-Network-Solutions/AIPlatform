<li>
    <a href="javascript:;" class="waves-effect"><i class="fa fa-history"></i>
        <span class="hide-menu">
            Audit Log
            <span class="fa arrow"></span>
        </span>
    </a>
    <ul class="nav nav-second-level">
        <li>
            <a href="{{route('admin.audit-log.log-activities')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.log_activities') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.attendance')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.attendanceLogs') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.leave')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.leaveLogs') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.incident')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.incidentLogs') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.user')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.userLogs') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.task')}}" class="waves-effect">
                <span class="hide-menu">{{ __('auditlog::app._log_activity.taskLogs') }}<span>
            </a>
        </li>
        <li>
            <a href="{{route('admin.audit-log.project')}}" class="waves-effect">
                <span class="hide-menu">Project Logs<span>
            </a>
        </li>
    </ul>
</li>
