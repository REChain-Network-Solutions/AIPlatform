<script>
    $('body').on('click', '#save-form', function() {
        $.easyAjax({
            url: "{{ route('einvoice.settings.save') }}",
            container: '#editSettings',
            type: "POST",
            redirect: true,
            file: true,
            data: $('#editSettings').serialize(),
            disableButton: true,
            blockUI: true,
            buttonSelector: "#save-form",
            success: function(response) {
                if (response.status == 'success') {
                    window.location.reload();
                }
            }
        })
    });
</script>
