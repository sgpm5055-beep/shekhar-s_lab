// Inter-agent context threading: accumulates outputs across pipeline steps

export class PipelineContext {
  private data: Record<string, string> = {};

  set(key: string, value: string) {
    this.data[key] = value;
  }

  get(key: string): string {
    return this.data[key] ?? "";
  }

  summary(): string {
    return Object.entries(this.data)
      .map(([k, v]) => `### ${k}\n${v}`)
      .join("\n\n");
  }
}
