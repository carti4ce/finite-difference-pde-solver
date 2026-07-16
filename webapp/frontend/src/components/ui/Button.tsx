import type { ButtonHTMLAttributes } from "react";

type Variant = "default" | "primary" | "ghost" | "danger";

export function Button({
  variant = "default",
  block,
  large,
  icon,
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  block?: boolean;
  large?: boolean;
  icon?: boolean;
}) {
  const classes = [
    "button",
    variant !== "default" ? variant : "",
    block ? "block" : "",
    large ? "large" : "",
    icon ? "icon" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...rest} />;
}
