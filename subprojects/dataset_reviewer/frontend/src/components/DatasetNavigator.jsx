export default function DatasetNavigator({
  filter,
  search,
  jump,
  disabled,
  onFilter,
  onSearch,
  onJumpChange,
  onJump,
  onPrevious,
  onNext,
  onUndo,
  onShortcuts,
}) {
  return (
    <section className="navigator-panel">
      <div className="navigator-group search-group">
        <label htmlFor="dataset-search">Search</label>
        <input
          id="dataset-search"
          type="search"
          disabled={disabled}
          value={search}
          placeholder="Search any field…"
          onChange={(event) => onSearch(event.target.value)}
        />
      </div>
      <div className="navigator-group">
        <label htmlFor="verdict-filter">View</label>
        <select id="verdict-filter" disabled={disabled} value={filter} onChange={(event) => onFilter(event.target.value)}>
          <option value="unreviewed">Unreviewed</option>
          <option value="all">All samples</option>
          <option value="reviewed">Reviewed</option>
          <option value="good">Good</option>
          <option value="bad">Bad</option>
          <option value="fix">Needs fix</option>
          <option value="unsure">Only unsure</option>
        </select>
      </div>
      <form className="navigator-group jump-group" onSubmit={onJump}>
        <label htmlFor="jump-index">Jump to</label>
        <div>
          <input
            id="jump-index"
            type="number"
            disabled={disabled}
            min="1"
            value={jump}
            placeholder="#"
            onChange={(event) => onJumpChange(event.target.value)}
          />
          <button type="submit" disabled={disabled}>Go</button>
        </div>
      </form>
      <div className="navigator-actions">
        <button type="button" disabled={disabled} onClick={onPrevious}>← Prev <kbd>K</kbd></button>
        <button type="button" disabled={disabled} onClick={onUndo}>Undo <kbd>Z</kbd></button>
        <button type="button" disabled={disabled} onClick={onNext}>Next <kbd>J</kbd> →</button>
        <button className="shortcut-button" type="button" onClick={onShortcuts}>?</button>
      </div>
    </section>
  );
}
