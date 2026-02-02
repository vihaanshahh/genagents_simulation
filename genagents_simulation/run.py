#!/usr/bin/env python
import argparse
import os
import json
import random
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from genagents_simulation.genagents.genagents import GenerativeAgent
from genagents_simulation.genagents.modules.interaction import categorical_resp_batch

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_CONFIG_PATH = "genagents_simulation/configs/llm_configs.json"

def load_llm_configs(config_path=LLM_CONFIG_PATH):
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return {config['config_name']: config for config in configs}
    except FileNotFoundError:
        logger.error(f"LLM config file not found at {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM config file at {config_path}")
        return {}

class BasicModule:
    def __init__(self, llm_config_name: str, agent_count: int, filters=None):
        self.llm_configs = load_llm_configs()
        
        # Extract LLM config name and agent count
        if not llm_config_name:
            raise ValueError("Missing 'llm_config_name' in input parameters.")
        if llm_config_name not in self.llm_configs:
            raise ValueError(f"LLM config '{llm_config_name}' not found in {LLM_CONFIG_PATH}")
        self.llm_config = self.llm_configs[llm_config_name]

        # Base paths for agents
        gss_base_path = os.path.join(os.path.dirname(__file__), "agent_bank/populations/gss_agents")
        single_agent_path = os.path.join(os.path.dirname(__file__), "agent_bank/populations/single_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")
        self.agents = []
        self.filter_stats = {}

        # Initialize agents based on count or percentage
        all_gss_agents = self._get_agent_folders(gss_base_path)
        total_agents_available = len(all_gss_agents)
        
        # Apply filters if provided
        if filters:
            self.filter_stats['filters_applied'] = filters
            self.filter_stats['total_agents_before_filter'] = total_agents_available
            all_gss_agents = self._filter_agents(all_gss_agents, filters)
            self.filter_stats['total_agents_after_filter'] = len(all_gss_agents)
            self.filter_stats['agents_filtered_out'] = total_agents_available - len(all_gss_agents)
            if not all_gss_agents:
                raise ValueError("No agents match the specified filters")
            logger.info(f"Filtered to {len(all_gss_agents)} agents matching criteria (filtered out {self.filter_stats['agents_filtered_out']})")
        else:
            self.filter_stats['filters_applied'] = None
            self.filter_stats['total_agents_available'] = total_agents_available
        
        if isinstance(agent_count, int):
            if agent_count == 1:
                selected_agents = [single_agent_path]
            else:
                selected_agents = random.sample(all_gss_agents, min(agent_count, len(all_gss_agents)))
        else:
            # Handle percentage-based agent_count if needed in the future
            percentage = int(agent_count.rstrip('%'))
            num_agents = max(1, int(len(all_gss_agents) * percentage / 100))
            selected_agents = random.sample(all_gss_agents, num_agents)

        self.filter_stats['agents_selected'] = len(selected_agents)
        self.filter_stats['agents_requested'] = agent_count

        for agent_path in selected_agents:
            try:
                agent = GenerativeAgent(agent_path)
                self.agents.append(agent)
            except Exception as e:
                logger.error(f"Failed to load agent: {str(e)}")

    def _get_agent_folders(self, base_path: str) -> List[str]:
        try:
            agent_folders = []
            for root, dirs, files in os.walk(base_path):
                if 'scratch.json' in files and 'meta.json' in files:
                    agent_folders.append(root)
            return sorted(agent_folders)
        except Exception as e:
            logger.error(f"Error accessing agent folders: {str(e)}")
            return []
    
    def _filter_agents(self, agent_paths: List[str], filters: dict) -> List[str]:
        """Filter agents based on demographic and location criteria."""
        import json
        filtered_paths = []
        
        logger.info(f"Filtering {len(agent_paths)} agents with filters: {filters}")
        
        for agent_path in agent_paths:
            try:
                scratch_path = os.path.join(agent_path, 'scratch.json')
                with open(scratch_path, 'r') as f:
                    agent_data = json.load(f)
                
                # Check each filter - only apply if filter exists and has values
                if filters.get('states') and len(filters['states']) > 0:
                    if agent_data.get('state') not in filters['states']:
                        continue
                
                if filters.get('cities') and len(filters['cities']) > 0:
                    if agent_data.get('city') not in filters['cities']:
                        continue
                
                if 'age_min' in filters and filters['age_min'] is not None:
                    if agent_data.get('age', 0) < filters['age_min']:
                        continue
                
                if 'age_max' in filters and filters['age_max'] is not None:
                    if agent_data.get('age', 999) > filters['age_max']:
                        continue
                
                if filters.get('sex') and len(filters['sex']) > 0:
                    if agent_data.get('sex') not in filters['sex']:
                        continue
                
                if filters.get('race') and len(filters['race']) > 0:
                    if agent_data.get('race') not in filters['race']:
                        continue
                
                if filters.get('political_views') and len(filters['political_views']) > 0:
                    if agent_data.get('political_views') not in filters['political_views']:
                        continue
                
                if filters.get('party_identification') and len(filters['party_identification']) > 0:
                    if agent_data.get('party_identification') not in filters['party_identification']:
                        continue
                
                if filters.get('religion') and len(filters['religion']) > 0:
                    if agent_data.get('religion') not in filters['religion']:
                        continue
                
                if filters.get('work_status') and len(filters['work_status']) > 0:
                    if agent_data.get('work_status') not in filters['work_status']:
                        continue
                
                if filters.get('marital_status') and len(filters['marital_status']) > 0:
                    if agent_data.get('marital_status') not in filters['marital_status']:
                        continue
                
                if filters.get('education') and len(filters['education']) > 0:
                    if agent_data.get('highest_degree_received') not in filters['education']:
                        continue
                
                filtered_paths.append(agent_path)
            except Exception as e:
                logger.error(f"Error filtering agent {agent_path}: {str(e)}")
                continue
        
        return filtered_paths

    def _process_single_agent(self, agent, input_data, progress_callback=None):
        try:
            agent_response = agent.categorical_resp(input_data)
            if not agent_response.get('responses') or len(agent_response['responses']) == 0:
                error = agent_response.get('error', 'Empty response from agent')
                logger.warning(f"Agent returned empty response: {error}")
                result = {
                    'responses': [],
                    'reasonings': [],
                    'error': error
                }
            else:
                result = agent_response
            
            result['agent_info'] = self._get_agent_demographics(agent)
            return result
        except Exception as e:
            logger.error(f"Agent processing error: {str(e)}")
            result = {
                'responses': [],
                'reasonings': [],
                'error': str(e),
                'agent_info': {}
            }
            return result

    def _process_agents_batch(self, agents, input_data, progress_callback=None, use_memory=None, llm_config=None):
        """Process all agents in a single batch for 100x speedup."""
        try:
            questions_list = [input_data] * len(agents)
            batch_results, optimization_settings = categorical_resp_batch(agents, questions_list, use_memory=use_memory, llm_config=llm_config)
            
            results = []
            for i, (agent, agent_response) in enumerate(zip(agents, batch_results)):
                if not agent_response.get('responses') or len(agent_response['responses']) == 0:
                    error = agent_response.get('error', 'Empty response from agent')
                    logger.warning(f"Agent {i} returned empty response: {error}")
                    result = {
                        'responses': [],
                        'reasonings': [],
                        'error': error
                    }
                else:
                    result = agent_response
                
                result['agent_info'] = self._get_agent_demographics(agent)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(agents), result)
            
            return results, optimization_settings
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return [{
                'responses': [],
                'reasonings': [],
                'error': str(e),
                'agent_info': {}
            }] * len(agents), None
    
    def _get_agent_demographics(self, agent):
        """Extract agent demographics for display."""
        scratch = agent.scratch
        return {
            'id': str(agent.id),
            'name': agent.get_fullname(),
            'age': scratch.get('age'),
            'sex': scratch.get('sex'),
            'ethnicity': scratch.get('ethnicity'),
            'race': scratch.get('race'),
            'city': scratch.get('city'),
            'state': scratch.get('state'),
            'street_address': scratch.get('street_address'),
            'political_views': scratch.get('political_views'),
            'party_identification': scratch.get('party_identification'),
            'religion': scratch.get('religion'),
            'work_status': scratch.get('work_status'),
            'marital_status': scratch.get('marital_status'),
            'highest_degree': scratch.get('highest_degree_received'),
            'total_wealth': scratch.get('total_wealth')
        }

    def func(self, input_data: Dict[str, List[str]], max_workers: int = None, progress_callback=None, cancel_event=None, use_memory=None):
        if max_workers is None:
            max_workers = min(len(self.agents), 10)
        
        logger.info(f"Running module function with {len(self.agents)} agents (parallel workers: {max_workers})")
        logger.debug(f"Input data received: {input_data}")

        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary with questions as keys and lists of options as values.")

        for question, options in input_data.items():
            if not isinstance(options, list):
                raise ValueError(f"Expected a list of options for question '{question}', but got {type(options).__name__}.")

        all_responses = []
        response_counts = {}
        explanations = {}

        for question, options in input_data.items():
            response_counts[question] = {}
            explanations[question] = []
            for option in options:
                response_counts[question][option] = 0

        if cancel_event and cancel_event.is_set():
            logger.info("Simulation cancelled by user")
            return {
                "individual_responses": [],
                "summary": {},
                "num_agents": len(self.agents),
            }
        
        logger.info(f"Processing {len(self.agents)} agents in batch mode for 100x speedup")
        
        optimization_settings = None
        try:
            batch_results, optimization_settings = self._process_agents_batch(self.agents, input_data, progress_callback, use_memory=use_memory, llm_config=self.llm_config)
            
            for idx, agent_response in enumerate(batch_results):
                if agent_response.get('responses') and len(agent_response['responses']) > 0:
                    all_responses.append(agent_response)
                    
                    for q_idx, question in enumerate(input_data):
                        if q_idx < len(agent_response['responses']):
                            response = agent_response['responses'][q_idx]
                            reasoning = agent_response['reasonings'][q_idx] if q_idx < len(agent_response.get('reasonings', [])) else ""
                            if response in response_counts[question]:
                                response_counts[question][response] += 1
                            else:
                                response_counts[question][response] = 1
                            explanations[question].append(reasoning)
                else:
                    all_responses.append(agent_response)
                    logger.warning(f"Agent {idx} returned empty response")
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}, falling back to individual processing")
            pending_agents = list(enumerate(self.agents))
            max_retries = 80
            retry_count = 0
            completed_count = 0
            
            while pending_agents and retry_count < max_retries:
                if cancel_event and cancel_event.is_set():
                    logger.info("Simulation cancelled by user")
                    break
                
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}: Processing {len(pending_agents)} failed agents")
                
                successful_indices = []
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index_agent = {
                        executor.submit(self._process_single_agent, agent, input_data): (idx, agent)
                        for idx, agent in pending_agents
                    }
                    
                    for future in as_completed(future_to_index_agent):
                        if cancel_event and cancel_event.is_set():
                            logger.info("Simulation cancelled during agent processing")
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except TypeError:
                                executor.shutdown(wait=False)
                            break
                        
                        idx, agent = future_to_index_agent[future]
                        try:
                            agent_response = future.result()
                            
                            completed_count += 1
                            if progress_callback:
                                progress_callback(completed_count, len(self.agents), agent_response)
                            
                            if agent_response.get('responses') and len(agent_response['responses']) > 0:
                                all_responses.append(agent_response)
                                successful_indices.append(idx)
                                
                                for q_idx, question in enumerate(input_data):
                                    if q_idx < len(agent_response['responses']):
                                        response = agent_response['responses'][q_idx]
                                        reasoning = agent_response['reasonings'][q_idx] if q_idx < len(agent_response.get('reasonings', [])) else ""
                                        if response in response_counts[question]:
                                            response_counts[question][response] += 1
                                        else:
                                            response_counts[question][response] = 1
                                        explanations[question].append(reasoning)
                            else:
                                logger.warning(f"Agent {idx} returned empty response, will retry")
                        except Exception as e:
                            logger.error(f"Agent {idx} failed with error: {str(e)}, will retry")
                
                if cancel_event and cancel_event.is_set():
                    break
                
                pending_agents = [(idx, agent) for idx, agent in pending_agents if idx not in successful_indices]
                retry_count += 1
                
                if pending_agents:
                    import time
                    wait_time = min(2.0 * retry_count, 15.0)
                    logger.info(f"Waiting {wait_time:.1f}s before retrying {len(pending_agents)} failed agents")
                    for _ in range(int(wait_time * 10)):
                        if cancel_event and cancel_event.is_set():
                            break
                        time.sleep(0.1)
            
            if pending_agents:
                logger.error(f"Failed to get responses from {len(pending_agents)} agents after {max_retries} retries")
                for idx, agent in pending_agents:
                    all_responses.append({
                        'responses': [],
                        'reasonings': [],
                        'error': f'Failed after {max_retries} retry attempts'
                    })

        visual_summary = {}
        for question in input_data:
            total = sum(response_counts[question].values())
            if total > 0:
                visual_summary[question] = {
                    'counts': response_counts[question],
                    'percentages': {option: f"{(count / total * 100):.1f}%" for option, count in response_counts[question].items()},
                    'visual': {option: f"{'█' * int(count / total * 20)} {count}/{total}" for option, count in response_counts[question].items()},
                    'explanations': explanations[question],
                }
            else:
                visual_summary[question] = {
                    'counts': response_counts[question],
                    'percentages': {},
                    'visual': {},
                    'explanations': explanations[question],
                }

        result = {
            "individual_responses": all_responses,
            "summary": visual_summary,
            "num_agents": len(self.agents),
        }
        
        # Add optimization settings if available
        if optimization_settings:
            result["optimization_settings"] = optimization_settings
        
        # Add filter stats if filters were applied
        if self.filter_stats:
            result["filter_stats"] = self.filter_stats
        
        return result

def run(llm_config_name: str, agent_count: int, func_name: str, func_input_data: Dict[str, List[str]], max_workers: int = 10, progress_callback=None, cancel_event=None, filters=None, use_memory=None):
    basic_module = BasicModule(llm_config_name, agent_count, filters=filters)
    method = getattr(basic_module, func_name, None)
    if method is None:
        raise ValueError(f"Method '{func_name}' not found in BasicModule")
    return method(func_input_data, max_workers=max_workers, progress_callback=progress_callback, cancel_event=cancel_event, use_memory=use_memory)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run agent simulations with custom questions and options.')
    parser.add_argument('--question', type=str, required=True, help='The question to ask the agents.')
    parser.add_argument('--options', type=str, required=True, help='Comma-separated options for the question (e.g., "Yes,No,Undecided").')
    parser.add_argument('--llm_config_name', type=str, default='gpt-5-nano', help='The LLM configuration name to use. Options: gpt-5-nano, gpt-oss-120b, claude-haiku-4.5, llama-3.2-3b, qwen3-next-80b, mistral-large-3, deepseek-r1')
    parser.add_argument('--agent_count', type=int, default=1, help='The number of agents to simulate.')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Process options into a list
    options_list = [option.strip() for option in args.options.split(',') if option.strip()]
    if not options_list:
        raise ValueError("At least one option must be provided.")

    # Prepare input data
    func_input_data = {
        args.question: options_list,
    }

    response = run(
        llm_config_name=args.llm_config_name,
        agent_count=args.agent_count,
        func_name="func",
        func_input_data=func_input_data
    )
    print("Response: ", response)
