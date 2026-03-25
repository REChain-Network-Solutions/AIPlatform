@php
$url = route('workorders.widgets.assignments', $work_order_id);
@endphp
<iframe src="{{ $url }}" style="width:100%;border:0;height:260px;" loading="lazy"></iframe>
