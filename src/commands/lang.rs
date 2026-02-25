use clap::Args;

use crate::{CLIResult, CommonArgs};

#[derive(Args, Debug)]
pub struct LangArgs {
    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: LangArgs) -> CLIResult<()> {
    dbg!(args);

    Ok(())
}
