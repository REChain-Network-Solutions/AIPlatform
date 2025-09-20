@if ($app_is_not_demo)
<script>
    document.addEventListener('DOMContentLoaded', function() {
        fetch('{{ route('dashboard.user.check.payment') }}')
            .then(response => response.json());
    });
</script>
@endif
