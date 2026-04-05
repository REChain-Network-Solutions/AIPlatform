<div class="col-md-12 mb-3">
    <div class="cm-summary-cards">
        @foreach(($doctorSummaryCards ?? []) as $card)
            <div class="cm-summary-card {{ ($card['theme'] ?? 'primary') === 'success' ? 'is-success' : '' }}">
                <h6>{{ $card['title'] }}</h6>
                <p>{{ $card['body'] }}</p>
            </div>
        @endforeach
    </div>
</div>
