@extends('feedback::layouts.master')

@section('content')
<div class="container py-4">\n@include('feedback::components.ai-insights')
  <h2>Customer Feedback & NPS</h2>

  <form method="post" action="{{ route('feedback.nps.store') }}" class="mb-4">
    @csrf
    <div class="row g-2">
      <div class="col-md-6">
        <input class="form-control" name="title" placeholder="Survey title" required>
      </div>
      <div class="col-md-6 d-flex align-items-center">
        <button class="btn btn-primary">Create NPS Survey</button>
      </div>
    </div>
  </form>

  <table class="table">
    <thead><tr><th>Survey</th><th>Promoters</th><th>Passives</th><th>Detractors</th><th>Total</th><th>NPS</th></tr></thead>
    <tbody>
      @foreach($surveys as $s)
        @php $a = $agg[$s->id] ?? null;
             $t = max(1, (int)($a->total ?? 0));
             $nps = round(((($a->promoters ?? 0)/$t) - (($a->detractors ?? 0)/$t)) * 100); @endphp
        <tr>
          <td>{{ $s->title }}</td>
          <td>{{ $a->promoters ?? 0 }}</td>
          <td>{{ $a->passives ?? 0 }}</td>
          <td>{{ $a->detractors ?? 0 }}</td>
          <td>{{ $a->total ?? 0 }}</td>
          <td>{{ $nps }}</td>
        </tr>
      @endforeach
    </tbody>
  </table>
</div>
@endsection
