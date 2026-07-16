# frozen_string_literal: true

module MobileWorldEmulatorClock
  OFFSET_SECONDS = Integer(ENV.fetch("MASTODON_TIME_OFFSET_SECONDS", "0"), exception: false) || 0

  def now
    super + OFFSET_SECONDS
  end
end

Time.singleton_class.prepend(MobileWorldEmulatorClock)
