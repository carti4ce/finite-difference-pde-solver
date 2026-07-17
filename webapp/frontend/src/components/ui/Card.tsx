import type { ReactNode } from "react";

export function Card({
  title,
  action,
  children,
}: {
  title?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="card">
      <div className="card-body">
        {title ? (
          <div className="card-title">
            <span>{title}</span>
            {action}
          </div>
        ) : null}
        {children}
      </div>
    </div>
  );
}
