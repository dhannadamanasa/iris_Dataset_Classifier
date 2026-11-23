"""
Defensive input validation, sanitization, and domain threshold warning engine.
"""
import re
from typing import Dict, List, Tuple
from src.config import FEATURE_CONSTRAINTS, CONFIDENCE_WARNING_THRESHOLD, ANOMALY_PETAL_SEPAL_RATIO, Colors
from src.models import ValidationResult


class IrisValidator:
    """Provides sanitization, boundary checks, and biological anomaly warnings."""

    @staticmethod
    def sanitize_input(raw_value: str) -> str:
        """Strips whitespace, surrounding quotes, and rogue characters."""
        if raw_value is None:
            return ""
        # Strip leading/trailing whitespaces, tabs, newlines, and single/double quotes
        cleaned = str(raw_value).strip().strip("\"'").replace(",", ".")
        return cleaned

    @classmethod
    def validate_feature(cls, feature_name: str, raw_value: str) -> ValidationResult:
        """
        Validates an individual botanical measurement against domain constraints.
        
        Rules:
        - Non-blank and non-empty
        - Numeric string parseable as float
        - Positive value (> 0)
        - Boundary limits (biological minimum & maximum)
        - Typical range alerts (warning if technically possible but unusual)
        """
        cleaned = cls.sanitize_input(raw_value)
        if not cleaned:
            return ValidationResult(
                is_valid=False,
                errors=[f"{feature_name}: Value cannot be blank or whitespace."]
            )

        # Regex check for numeric format (allows standard decimal representation)
        if not re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", cleaned):
            return ValidationResult(
                is_valid=False,
                errors=[f"{feature_name}: '{raw_value}' is not a valid numeric value."]
            )

        try:
            val = float(cleaned)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                errors=[f"{feature_name}: Unable to parse '{raw_value}' as a numeric float."]
            )

        if val <= 0:
            return ValidationResult(
                is_valid=False,
                errors=[f"{feature_name}: Value must be strictly positive (> 0), got {val}."]
            )

        constraints = FEATURE_CONSTRAINTS.get(feature_name)
        if not constraints:
            return ValidationResult(is_valid=True, cleaned_value=round(val, 2))

        errors: List[str] = []
        warnings: List[str] = []

        # Check biological hard extremes (values that are biologically impossible for Iris flowers)
        extreme_min = constraints["extreme_min"]
        extreme_max = constraints["extreme_max"]
        if val < extreme_min or val > extreme_max:
            errors.append(
                f"{constraints['name']}: {val} {constraints['unit']} breaches biological hard limits "
                f"[{extreme_min} - {extreme_max} {constraints['unit']}]."
            )
            return ValidationResult(is_valid=False, errors=errors)

        # Check standard operational botanical ranges (warn if unusual outlier)
        std_min = constraints["min"]
        std_max = constraints["max"]
        if val < std_min:
            warnings.append(
                f"Boundary Alert: {constraints['name']} ({val} {constraints['unit']}) is unusually small "
                f"(typical min is {std_min} {constraints['unit']})."
            )
        elif val > std_max:
            warnings.append(
                f"Boundary Alert: {constraints['name']} ({val} {constraints['unit']}) is unusually large "
                f"(typical max is {std_max} {constraints['unit']})."
            )

        return ValidationResult(
            is_valid=True,
            cleaned_value=round(val, 2),
            errors=[],
            warnings=warnings
        )

    @classmethod
    def validate_sample_set(
        cls,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validates all 4 features together and checks cross-feature botanical anomalies.
        Returns: (is_valid, list_of_errors, list_of_warnings)
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []

        mapping = {
            "sepal_length": str(sepal_length),
            "sepal_width": str(sepal_width),
            "petal_length": str(petal_length),
            "petal_width": str(petal_width)
        }

        cleaned_vals = {}
        for feat, raw in mapping.items():
            res = cls.validate_feature(feat, raw)
            if not res.is_valid:
                all_errors.extend(res.errors)
            else:
                cleaned_vals[feat] = res.cleaned_value
                all_warnings.extend(res.warnings)

        if all_errors:
            return False, all_errors, all_warnings

        # Cross-feature morphological anomaly checks:
        sl = cleaned_vals["sepal_length"]
        sw = cleaned_vals["sepal_width"]
        pl = cleaned_vals["petal_length"]
        pw = cleaned_vals["petal_width"]

        # Anomaly 1: In Iris flowers, petal length is almost never greater than sepal length
        if pl > (sl * ANOMALY_PETAL_SEPAL_RATIO):
            all_warnings.append(
                f"Morphological Anomaly: Petal length ({pl} cm) exceeds Sepal length ({sl} cm). "
                f"This ratio is rare in natural Iris specimens."
            )

        # Anomaly 2: Petal width exceeding petal length
        if pw > pl:
            all_warnings.append(
                f"Morphological Anomaly: Petal width ({pw} cm) is greater than Petal length ({pl} cm)."
            )

        # Anomaly 3: Slenderness ratio (Sepal length / Sepal width)
        if sw > 0 and (sl / sw) < 0.9:
            all_warnings.append(
                f"Morphological Anomaly: Sepal appears wider than it is long ({sl} cm vs {sw} cm)."
            )

        return True, [], all_warnings

    @staticmethod
    def evaluate_prediction_warnings(
        confidence: float,
        existing_warnings: List[str]
    ) -> List[str]:
        """Appends threshold warnings for model predictions (e.g., low confidence)."""
        warnings = list(existing_warnings)
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            pct = round(confidence * 100, 1)
            warnings.append(
                f"Low Confidence Flag: Model certainty is {pct}% (below {int(CONFIDENCE_WARNING_THRESHOLD * 100)}% threshold). "
                f"Specimen lies near classification decision boundary."
            )
        return warnings


def prompt_validated_float(
    feature_name: str,
    prompt_text: Optional[str] = None,
    allow_default: Optional[float] = None
) -> float:
    """
    Centralized terminal input wrapper that loops until valid numeric entry is provided.
    Implements tip from section 8 of hackathon guide.
    """
    constraints = FEATURE_CONSTRAINTS[feature_name]
    default_hint = f" [default: {allow_default}]" if allow_default is not None else ""
    display_prompt = prompt_text or f"Enter {constraints['name']} ({constraints['min']} - {constraints['max']} cm){default_hint}: "

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.CYAN}{display_prompt}{Colors.RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
            raise

        if not user_input.strip() and allow_default is not None:
            return allow_default

        result = IrisValidator.validate_feature(feature_name, user_input)
        if result.is_valid:
            if result.warnings:
                for w in result.warnings:
                    print(f"  {Colors.YELLOW}⚠️  {w}{Colors.RESET}")
            return result.cleaned_value
        else:
            for err in result.errors:
                print(f"  {Colors.RED}❌ {err}{Colors.RESET}")
            print(f"  {Colors.DIM}Please re-enter a valid measurement.{Colors.RESET}")
