from typing import Any, Dict, List, Mapping, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, PrivateAttr, SecretStr, model_validator

from langchain_community.llms.utils import enforce_stop_tokens

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_QUALITY_PRESET = "medium"
DEFAULT_MAX_TOKENS = 512


class TrustworthyLanguageModel(BaseLLM):
    """Cleanlab's Trustworthy Language Model (TLM).

    To use, you should have the ``cleanlab-studio`` python package installed,
    and the API key set either in the ``CLEANLAB_API_KEY`` environment variable,
    or pass it as a named parameter to the constructor.
    Sign up at app.cleanlab.ai to get a free API key.

    Example:
        .. code-block:: python

            from langchain_community.llms import TrustworthyLanguageModel
            tlm = TrustworthyLanguageModel(
                cleanlab_api_key="my_api_key",  # Optional if `CLEANLAB_API_KEY` is set
                quality_preset="best"
            )
    """

    _client: Any = PrivateAttr()  # :meta private:

    cleanlab_api_key: Optional[SecretStr] = Field(default=None)
    """Cleanlab API key. Get it here: https://app.cleanlab.ai"""

    quality_preset: str = Field(default=DEFAULT_QUALITY_PRESET)
    """Presets to vary the quality of LLM response. Available presets listed here: 
        https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions
    """

    options: Dict[str, Any] = Field(default_factory=dict)
    """Holds configurations for trustworthy language model. 
       Available options (model, max_tokens, etc.) with their definitions listed here: 
       https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions
    """

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cleanlab_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "cleanlab_api_key", "CLEANLAB_API_KEY")
        )
        values["cleanlab_api_key"] = cleanlab_api_key

        try:
            from cleanlab_studio import Studio

            studio = Studio(api_key=cleanlab_api_key.get_secret_value())
            # Check for user overrides in options dict
            use_options = values["options"] is not None
            # Initialize TLM
            cls._client = studio.TLM(
                quality_preset=values["quality_preset"],
                options=values["options"] if use_options else None,
            )
        except ImportError:
            raise ImportError(
                "Could not import cleanlab-studio python package. "
                "Please install it with `pip install -U cleanlab-studio`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"quality_preset": self.quality_preset},
            **{"options": self.options},
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cleanlab API."""
        default_params = {
            "quality_preset": DEFAULT_QUALITY_PRESET,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "model": DEFAULT_MODEL,
        }
        return {**default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "trustworthy_language_model"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call Cleanlab endpoint and return response with additional info."""

        responses: List[Dict[str, Any]] = self._client.prompt(prompts)

        generations = []
        for resp in responses:
            text = resp["response"]
            trustworthiness_score = resp["trustworthiness_score"]

            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            generations.append(
                [
                    Generation(
                        text=text,
                        generation_info={
                            "trustworthiness_score": trustworthiness_score,
                            **(
                                {"explanation": resp["log"]["explanation"]}
                                if "explanation" in self.options.get("log", [])
                                else {}
                            ),
                        },
                    )
                ]
            )

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously call to Cleanlab endpoint."""

        responses: List[Dict[str, Any]] = await self._client.prompt_async(prompts)

        generations = []
        for resp in responses:
            text = resp["response"]
            trustworthiness_score = resp["trustworthiness_score"]
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            generations.append(
                [
                    Generation(
                        text=text,
                        generation_info={
                            "trustworthiness_score": trustworthiness_score,
                            **(
                                {"explanation": resp["log"]["explanation"]}
                                if "explanation" in self.options.get("log", [])
                                else {}
                            ),
                        },
                    )
                ]
            )

        return LLMResult(generations=generations)

    def get_trustworthiness_score(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Calculate trustworthiness scores for prompt-response pairs.

        Uses TrustworthyLanguageModel to evaluate the reliability of responses
        by generating a trustworthiness score between 0 and 1. Higher scores
        indicate more reliable responses.

        Args:
            prompts: Input prompt(s) to evaluate
            responses: Response(s) to analyze

        Returns:
            Union[float, List[float]]: Trustworthiness score(s) between 0 and 1
                - float: For single prompt-response pair
                - List[float]: For multiple prompt-response pairs

        Examples:
            >>> prompt = "Explain the process of photosynthesis"
            >>> response = (
            ...     "Photosynthesis is how plants convert sunlight into energy. "
            ...     "They use chlorophyll to capture sunlight and combine it with "
            ...     "CO2 and water to produce glucose and oxygen."
            ... )
            >>> get_trustworthiness_score(prompt, response)
            0.92
        """

        # Convert to list
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        response_list = [responses] if isinstance(responses, str) else responses

        tlm_response = self._client.try_get_trustworthiness_score(
            prompt_list, response_list
        )
        scores = [resp["trustworthiness_score"] for resp in tlm_response]

        return scores[0] if isinstance(prompts, str) else scores
