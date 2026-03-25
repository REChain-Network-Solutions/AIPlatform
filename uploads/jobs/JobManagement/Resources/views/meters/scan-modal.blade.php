<div class="modal-header">
    <h5 class="modal-title">@lang('engineerings::modules.meter') @lang('app.details')</h5>
    <button type="button" class="close" data-dismiss="modal" aria-hidden="true">×</button>
</div>
<div class="modal-body">
    <div class="portlet-body">
        <div class="row">
            <div class="col">
              <div id="reader"></div>
            </div>
            <div class="col" style="padding: 30px">
              <h4>Scan Result </h4>
              <div id="result">
                Result goes here
              </div>
            </div>
          
          </div>
    </div>
</div>
<div class="modal-footer">
    <x-forms.button-cancel data-dismiss="modal" class="border-0 mr-3">@lang('app.cancel')</x-forms.button-cancel>
</div>

<script></script>
