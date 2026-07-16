import type { TimeOrder } from "../../api/types";
import { Card } from "../ui/Card";
import { SegmentedControl } from "../ui/SegmentedControl";

export function TimeOrderSection({
  value,
  onChange,
}: {
  value: TimeOrder;
  onChange: (v: TimeOrder) => void;
}) {
  return (
    <Card title="Equation">
      <SegmentedControl
        value={value}
        onChange={onChange}
        options={[
          { value: 0, label: "0 = f(u)" },
          { value: 1, label: "uₜ = f(u)" },
          { value: 2, label: "uₜₜ = f(u)" },
        ]}
      />
    </Card>
  );
}
