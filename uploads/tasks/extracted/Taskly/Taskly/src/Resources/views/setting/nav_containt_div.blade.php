<div class="tab-pane fade" id="customer-project" role="tabpanel" aria-labelledby="pills-user-tab-4">
    <div class="row">
        <div class="col-sm-12">
            <div class="card">
                <div class="card-body table-border-style">
                    <div class="table-responsive">
                        <table class="table mb-0 pc-dt-simple" id="customer_project">
                            <thead>
                                <tr>
                                    <th>{{ __('Name') }}</th>
                                    <th>{{ __('Stage') }}</th>
                                    <th>{{ __('Start Date') }}</th>
                                    <th>{{ __('End Date') }}</th>
                                    <th>{{ __('description') }}</th>
                                    @if(Laratrust::hasPermission('project show') || Laratrust::hasPermission('project edit') || Laratrust::hasPermission('project delete'))
                                        <th width="10%"> {{ __('Action') }}</th>
                                    @endif
                                </tr>
                            </thead>
                            <tbody>
                                @forelse (\Workdo\Taskly\Entities\Project::customerProject($customer->id) as $client_project)
                                <tr class="font-style">
                                    <td>{{ !empty($client_project->project) ? $client_project->project->name : '' }}</td>
                                    <td>{{ !empty($client_project->project) ? $client_project->project->status : '' }}</td>
                                    <td>{{ !empty($client_project->project) ? company_date_formate($client_project->project->start_date) : ''}}</td>
                                    <td>{{ !empty($client_project->project) ? company_date_formate($client_project->project->end_date) : '' }}</td>
                                    <td>
                                        <p style="white-space: nowrap;
                                            width: 200px;
                                            overflow: hidden;
                                            text-overflow: ellipsis;">{{ !empty($client_project->project) ? $client_project->project->description  : ''}}
                                        </p>
                                    </td>
                                    @if (Laratrust::hasPermission('project show'))
                                        <td class="Action">
                                            <span>
                                                <div class="action-btn">
                                                    <a href="{{ route('projects.show',$client_project->project_id) }}" data-bs-toggle="tooltip" title="{{__('Details')}}"  data-title="{{__('Project Details')}}" class="mx-3 btn btn-sm align-items-center bg-warning">
                                                        <i class="ti ti-eye text-white"></i>
                                                    </a>
                                                </div>
                                            </span>
                                        </td>
                                    @endif
                                </tr>
                                    @empty
                                    @include('layouts.nodatafound')
                                @endforelse
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
