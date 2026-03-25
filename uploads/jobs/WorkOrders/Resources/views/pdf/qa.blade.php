<!doctype html><html><head><meta charset='utf-8'>
<style>@page{margin:80px 40px 70px 40px;}header{position:fixed;top:-60px;left:0;right:0;height:50px;font-size:12px;}
footer{position:fixed;bottom:-50px;left:0;right:0;height:40px;font-size:11px;color:#666;}table{width:100%;border-collapse:collapse;font-size:12px;}
th,td{border:1px solid #ddd;padding:6px;}.badge{padding:2px 6px;border:1px solid #ccc;border-radius:4px;font-size:10px;}</style></head>
<body>
<header><table style='border:0'><tr>
<td style='width:30%;border:0'>@if($brand['logo'])<img src='{{ public_path($brand['logo']) }}' style='max-height:45px'>@endif</td>
<td style='text-align:right;border:0'>{{ $brand['header'] }}</td></tr></table></header>
<footer><div>{{ $brand['footer'] }}</div></footer>
<h3 style='margin-top:0'>@term('work_order') #{{ $wo->id }} — QA & Completion</h3>
<table>
<tr><th style='width:30%'>Client</th><td>{{ $wo->client_name ?? ('Client #'.$wo->client_id) }}</td></tr>
<tr><th>Scheduled</th><td>{{ $wo->scheduled_at }}</td></tr>
<tr><th>Status</th><td>{{ $wo->status }}</td></tr>
</table>
@if($wc)
<h4 style='margin-top:16px'>Checklist</h4>
<table><thead><tr><th style='width:65%'>Item</th><th style='width:15%'>Status</th><th>Notes</th></tr></thead><tbody>
@foreach($items as $i)
<tr><td>{{ $i->text }}</td><td>
@switch($i->status)@case('pass')<span class='badge'>PASS</span>@break
@case('fail')<span class='badge'>FAIL</span>@break
@case('na')<span class='badge'>N/A</span>@break
@default<span class='badge'>PENDING</span>@endswitch
</td><td>{{ $i->notes }}</td></tr>
@endforeach
</tbody></table>
@endif
@if($wo->client_signed_at)
<h4 style='margin-top:16px'>Client Sign-off</h4>
<table>
<tr><th>Name</th><td>{{ $wo->client_sign_name }}</td></tr>
<tr><th>Signed at</th><td>{{ $wo->client_signed_at }}</td></tr>
<tr><th>Signature</th><td>@if($wo->client_signature_path)<img src='{{ public_path($wo->client_signature_path) }}' style='height:80px'>@endif</td></tr>
</table>
@endif
</body></html>